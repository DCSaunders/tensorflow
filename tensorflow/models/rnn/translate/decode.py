"""Binary for decoding from translation models based on tensorflow/models/rnn/translate/translate.py.

Note that this decoder is greedy and very basic. For a better decoder, see http://ucam-smt.github.io/sgnmt/html/tutorial.html
which supports decoding from tensorflow models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import sys
import numpy as np
import datetime
import logging

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

if __name__ == "__main__":
  from tensorflow.models.rnn.translate.train import FLAGS as train_flags # get default model settings from train.py
from tensorflow.models.rnn.translate.utils import data_utils, model_utils

# Decoder settings
tf.app.flags.DEFINE_string("test_src_idx", "/tmp/in.txt", "An integer-encoded input file")
tf.app.flags.DEFINE_string("test_out_idx", "/tmp/out.txt", "Output file for decoder output")
tf.app.flags.DEFINE_string("test_hidden_state", "/tmp/hidden.txt", "Output file for hidden state")
tf.app.flags.DEFINE_integer("max_sentences", 0, "The maximum number of sentences to translate (all if set to 0)")
tf.app.flags.DEFINE_string("decode_hidden", None, "Decode from hidden layers in file")
tf.app.flags.DEFINE_boolean("interactive", False, "Decode from command line")
tf.app.flags.DEFINE_string("decode_interpolate_hidden", None, "Decode from hidden layers interpolating between true and generated hidden vectors")
tf.app.flags.DEFINE_string("model_path", None, "Path to trained model")
FLAGS = tf.app.flags.FLAGS

def decode(config, input=None, output=None, max_sentences=0):
  if input and output:
    inp = input
    out = output
  else:
    inp = config['test_src_idx']
    out = config['test_out_idx']
  if 'test_hidden_state' in config:
    hidden = config['test_hidden_state']
  else:
    hidden = '/tmp/hidden'
  
  max_sents = 0
  if 'max_sentences' in config:
    max_sents = config and config['max_sentences']
  if max_sentences > 0:
    max_sents = max_sentences
  if config['decode_hidden']:
    unpickle_hidden(config, out, max_sentences=max_sents)
  elif config['decode_interpolate_hidden']:
    decode_interpolate_hidden(config, out, max_sentences=max_sents)
  else:
    # Find longest input to create suitable bucket
    max_input_length = 0
    with open(inp) as f_in:
      for sentence in f_in:
        token_ids = [ int(tok) for tok in sentence.strip().split() ]
        if config['add_src_eos']:
          token_ids.append(data_utils.EOS_ID)
        if len(token_ids) > max_input_length:
          max_input_length = len(token_ids)
    bucket = model_utils.make_bucket(max_input_length, greedy_decoder=True)
    buckets = list(model_utils._buckets)
    buckets.append(bucket)
    logging.info("Add new bucket={}".format(bucket))

    with tf.Session() as session:
      # Create model and load parameters: uses the training graph for decoding
      model = model_utils.create_model(session, config, forward_only=True,
                                          buckets=buckets)
      model.batch_size = 1  # We decode one sentence at a time.

      # Decode input file
      num_sentences = 0
      logging.info("Start decoding, max_sentences=%i" % max_sents)
      with open(inp) as f_in, open(out, 'w') as f_out, open(hidden, 'wb') as f_hidden:
        pickler = cPickle.Pickler(f_hidden)
        for sentence in f_in:
          outputs, states = get_outputs(session, config, model, sentence, buckets)
          logging.info("Output: {}".format(outputs))
          # If there is an EOS symbol in outputs, cut them at that point.
          if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
          print(" ".join([str(tok) for tok in outputs]), file=f_out)
          pickler.dump({'states': states, 'length': len(outputs)})
          num_sentences += 1
          if max_sents > 0 and num_sentences >= max_sents:
            break
  logging.info("Decoding completed.")

def unpickle_hidden(config, out, max_sentences=0):
  hidden_list = []
  with open(config['decode_hidden'], 'rb') as f_in:
    unpickler = cPickle.Unpickler(f_in)
    hidden_size = config['hidden_size']
    while True and (max_sentences == 0 or len(hidden_list) < max_sentences):
      try:
        hidden = np.array(unpickler.load()['states'])
        hidden = hidden.reshape(1, 2*hidden_size) # batch_size, BiRNN size
        hidden_list.append(hidden)
      except (EOFError):
        break
  with tf.Session() as session:
    model = model_utils.create_model(session, config, forward_only=True, hidden=True)
    model.batch_size = 1  # We decode one sentence at a time.

    decode_hidden(session, model, config, out, hidden_list)

def decode_interpolate_hidden(config, out, max_sentences=0):
  hidden_list = []
  hidden_size = config['hidden_size']
  num_decoded = 0
  with tf.Session() as session:
    model = model_utils.create_model(session, config, forward_only=True, hidden=True)
    model.batch_size = 1  # We decode one sentence at a time.

    with open(config['decode_interpolate_hidden'], 'rb') as f_in:
      label_samples = cPickle.load(f_in)
      for label in label_samples:
        log_msg(config['test_out_idx'],
                'Gaussian mixture component: {}\n'.format(label))
        for interp_list in label_samples[label]:
          log_msg(config['test_out_idx'], 'New interpolation set\n')
          for i in range(0, len(interp_list)):
            interp_list[i] = interp_list[i].reshape(1, 2*hidden_size)
          decode_hidden(session, model, config, out, interp_list, append=True)
          num_decoded += 1
          if num_decoded > max_sentences and max_sentences > 0:
            break
     
def log_msg(f_name, msg):
  with open(f_name, 'a') as f_out:
    f_out.write(msg)

def decode_hidden(session, model, config, out, hidden_list, append=False):
  if append:
    mode = 'a'
  else:
    mode = 'w'
  with open(config['test_out_idx'], mode) as f_out:
    for hidden in hidden_list:
      #hidden = np.random.randn(1, 1000)
      outputs, states = get_outputs(session, config, model, sentence='', hidden=hidden)
      logging.info("Output: {}".format(outputs))
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      print(" ".join([str(tok) for tok in outputs]), file=f_out)


def decode_interactive(config):
  with tf.Session() as session:
    # Create model and load parameters: uses the training graph for decoding
    config['batch_size'] = 1 # We decode one sentence at a time.
    model = model_utils.create_model(session, config, forward_only=True)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      outputs, states = get_outputs(session, config, model, sentence)
      print("Output: %s" % " ".join([str(tok) for tok in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

def get_outputs(session, config, model, sentence, buckets=None, hidden=None):
  # Get token-ids for the input sentence.
  token_ids = [ int(tok) for tok in sentence.strip().split() ]
  token_ids = [ w if w < config['src_vocab_size'] else data_utils.UNK_ID
                for w in token_ids ]
  if config['add_src_eos']:
    token_ids.append(data_utils.EOS_ID)

  if not buckets:
    buckets = model_utils._buckets
  if hidden is None:
    bucket_id = min([b for b in xrange(len(buckets))
                     if buckets[b][0] >= len(token_ids)])
  else:
    bucket_id = max([b for b in xrange(len(buckets))])
  logging.info("Bucket {}".format(buckets[bucket_id]))
  logging.info("Input: {}".format(token_ids))

  # Get a 1-element batch to feed the sentence to the model.
  encoder_inputs, decoder_inputs, target_weights, sequence_length, src_mask, bow_mask = model.get_batch(
    {bucket_id: [(token_ids, [])]}, bucket_id, config['encoder'])

  # Get output logits for the sentence.
  _, _, output_logits, hidden_states = model.get_state_step(
    session, encoder_inputs, 
    decoder_inputs,
    target_weights, bucket_id, 
    forward_only=True,
    sequence_length=sequence_length,
    src_mask=src_mask, bow_mask=bow_mask, hidden=hidden)

  # This is a greedy decoder - outputs are just argmaxes of output_logits.
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
  return outputs, hidden_states

def main(_):
  config = model_utils.process_args(FLAGS, train=False, greedy_decoder=True)
  if FLAGS.interactive:
    decode_interactive(config)
  else:
    decode(config)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  logging.info("Start: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))
  tf.app.run()
  logging.info("End: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))

