# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate.utils import data_utils
from tensorflow.models.rnn.translate.seq2seq.wrapper_cells import BidirectionalRNNCell, BOWCell

import logging
from tensorflow.core.protobuf import saver_pb2

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """
  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               embedding_size,
               hidden_size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               use_lstm=False,
               num_samples=512,
               forward_only=False,
               dtype=tf.float32,
               opt_algorithm="sgd",
               encoder="reverse",
               use_sequence_length=False,
               use_src_mask=False,
               maxout_layer=False,
               init_backward=False,
               no_pad_symbol=False,
               variable_prefix=None,
               rename_variable_prefix=None,
               init_const=False,
               use_bow_mask=False,
               max_to_keep=0,
               keep_prob=1.0,
               initializer=None,
               legacy=False,
               hidden=False,
               latent_size=None,
               annealing=False,
               anneal_steps=1000,
               word_keep_prob=1.0,
               scheduled_sample=True,
               scheduled_sample_steps=1000,
               kl_min=0.0,
               sample_mean=False,
               concat_encoded=False,
               seq2seq_mode="nmt",
               bow_no_replace=False,
               grammar=None,
               mean_kl=False,
               single_graph=False):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    with tf.variable_scope(variable_prefix or ""):
      self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
      self.global_step = tf.Variable(0, trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.no_pad_symbol = no_pad_symbol
    self.seq2seq_mode = seq2seq_mode
    self.annealing = annealing
    self.anneal_steps = anneal_steps
    self.scheduled_sample = scheduled_sample
    self.scheduled_sample_steps = scheduled_sample_steps
    self.word_keep_prob = word_keep_prob
    self.bow_no_replace = bow_no_replace
    self.anneal_scale = tf.placeholder(tf.float32, shape=[])
    self.single_graph = single_graph
    self.grammar = grammar
    if self.grammar:
      self.grammar.stack_nops = self.buckets[-1][1]
      self.grammar.batch_size = self.batch_size

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, hidden_size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
      logging.info("Using output projection of shape (%d, %d)" % (hidden_size, self.target_vocab_size))
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(inputs, tf.float32)
        return tf.cast(
            tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                       num_samples, self.target_vocab_size),
            dtype)
      softmax_loss_function = sampled_loss
    else:
      logging.info("Using maxout_layer=%r and full softmax loss" % maxout_layer)
    
    def get_cell(n_hidden):
      logging.info("Constructing cell of size={}".format(n_hidden))
      if use_lstm:
        logging.info("Using LSTM cells")
        if initializer:
          single_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, initializer=initializer)
        else:
          # to use peephole connections, cell clipping or a projection layer, use LSTMCell
          single_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
      else:
        logging.info("Using GRU cells")
        single_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
      cell = single_cell
      if not forward_only and use_lstm and keep_prob < 1:
        logging.info("Adding dropout wrapper around lstm cells")
        single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=keep_prob)
      if encoder == "bidirectional":
        logging.info("Bidirectional model")
        if init_backward:
          logging.info("Use backward encoder state to initialize decoder state")
        cell = BidirectionalRNNCell([single_cell] * 2)
      elif encoder == "bow":
        logging.info("BOW model")
        if num_layers > 1:
          logging.info("Model with %d layers for the decoder" % num_layers)
          cell = BOWCell(tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers))
        else:
          cell = BOWCell(single_cell)
      elif num_layers > 1:
        logging.info("Model with %d layers" % num_layers)
        cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
      return cell

    cell = get_cell(hidden_size)
    dec_cell = cell
    if concat_encoded and self.seq2seq_mode == 'vae':
      dec_cell = get_cell(hidden_size + latent_size)

    # The seq2seq function: we use embedding for the input and attention.
    logging.info("Embedding size={}".format(embedding_size))
    scope = None
    if variable_prefix is not None:
      if self.seq2seq_mode in ('autoencoder', 'vae'):
        if legacy:
          scope = variable_prefix+"/embedding_tied_rnn_seq2seq"
        else:
          scope = variable_prefix+"/embedding_rnn_seq2seq"
      else:
        scope = variable_prefix+"/embedding_attention_seq2seq"
      logging.info("Using variable scope {}".format(scope)) 
      

    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode, bucket_length, encoder_state=None,
                  feed_prev_p=None):
      seq2seq_args = dict(encoder_inputs=encoder_inputs,
                          decoder_inputs=decoder_inputs,
                          cell=cell, 
                          embedding_size=embedding_size,
                          output_projection=output_projection,
                          feed_previous=do_decode,
                          dtype=dtype,
                          encoder=encoder,
                          sequence_length=self.sequence_length,
                          bucket_length=bucket_length,
                          init_backward=init_backward,
                          scope=scope,
                          legacy=legacy,
                          bow_mask=self.bow_mask)
      if self.seq2seq_mode in ('autoencoder', 'vae'):
        seq2seq_args.update(num_symbols=source_vocab_size,
                            feed_prev_p=feed_prev_p,
                            hidden_state=encoder_state,
                            bow_no_replace=bow_no_replace,
                            grammar=self.grammar)
        if self.seq2seq_mode == 'vae':
          logging.info("Creating embedding rnn variational autoencoder")
          seq2seq_args.update(latent_size=latent_size,
                              transfer_func=tf.nn.relu,
                              anneal_scale=self.anneal_scale,
                              kl_min=kl_min,
                              sample_mean=sample_mean,
                              concat_encoded=concat_encoded,
                              dec_cell=dec_cell,
                              mean_kl=mean_kl)
          return tf.nn.seq2seq.embedding_rnn_vae_seq2seq(**seq2seq_args)
        else:
          logging.info("Creating embedding rnn autoencoder")
          return tf.nn.seq2seq.embedding_rnn_autoencoder_seq2seq(**seq2seq_args)
      else:
        logging.info('Creating embedding attention model')
        seq2seq_args.update(num_encoder_symbols=source_vocab_size,
                            num_decoder_symbols=target_vocab_size,
                            src_mask=self.src_mask,
                            maxout_layer=maxout_layer,
                            bow_emb_size=hidden_size,
                            init_const=init_const,
                            keep_prob=keep_prob)
        return tf.nn.seq2seq.embedding_attention_seq2seq(**seq2seq_args)
                    
    # Feeds for inputs.
    self.encoder_inputs = []
    enc_state_size = 2 * hidden_size if use_lstm else hidden_size
    self.encoder_states = tf.placeholder(tf.float32, shape=[None, enc_state_size])
    self.decoder_inputs = []
    self.feed_prev_p = tf.placeholder(tf.float32, shape=[])
    self.target_weights = []
    self.targets = []
    if self.grammar:
      self.grammar.grammar_mask = []
      self.grammar.rhs_mask = tf.placeholder(
        tf.int32, shape=[None, None], name='grammar_rhs')
      if forward_only:
        self.grammar.grammar_full_mask = tf.placeholder(
          tf.float32, shape=[None, None], name='grammar_mask')
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))                                                      
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                name="weight{0}".format(i)))
      self.targets.append(tf.placeholder(tf.int32, shape=[None],
                                         name="target{0}".format(i)))
      if self.grammar is not None:
        self.grammar.grammar_mask.append(tf.placeholder(tf.float32, shape=[None, None],
                                                name='grammar{0}'.format(i)))

    if use_sequence_length is True:
      logging.info("Using sequence length for encoder")                          
      self.sequence_length = tf.placeholder(tf.int32, shape=[None], name="seq_len")
    else:
      self.sequence_length = None
      
    if use_src_mask:
      logging.info("Using source mask for decoder")
      self.src_mask = tf.placeholder(tf.float32, shape=[None, None],
                                     name="src_mask")
    else:
      self.src_mask = None

    if use_bow_mask:
      logging.info("Using bow mask for output layer")
      self.bow_mask = tf.placeholder(tf.float32, shape=[None, None],
                                     name="bow_mask")
    else:
      self.bow_mask = None
      

    # Training outputs and losses.
    def adjust_loss_vae():
      self.reconstruct_loss = [tf.placeholder(tf.float32, shape=[]) for _ in range(len(self.losses))]
      self.kl_loss = [tf.placeholder(tf.float32, shape=[]) for _ in range(len(self.losses))]
      for b in range(len(self.losses)):
        self.reconstruct_loss[b] = self.losses[b][0]
        self.kl_loss[b] = self.losses[b][1] 
        self.losses[b] = tf.add(self.reconstruct_loss[b], self.kl_loss[b])
    state = self.encoder_states if hidden else None
    feed_prev = self.feed_prev_p if (self.scheduled_sample and not forward_only) else None
    model_args = dict(encoder_inputs=self.encoder_inputs, decoder_inputs=self.decoder_inputs, 
                      targets=self.targets, weights=self.target_weights, buckets=buckets, 
                      seq2seq=lambda a, b, c, d, e: seq2seq_f(a, b, False, c, d, e),
                      softmax_loss_function=softmax_loss_function,
                      encoder_states=state, feed_prev_p=feed_prev)
    if self.single_graph:
      model_args.update(single_graph=self.single_graph)
    if forward_only:
      model_args.update(seq2seq=lambda a, b, c, d, e: seq2seq_f(a, b, True, c, d, e))
      self.outputs, self.losses, self.states = tf.nn.seq2seq.model_with_buckets_states(**model_args)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          # Similar to what is done in the loop function (where xw_plus_b is used instead of matmul).
          # The loop function also takes the argmax, but the result is not saved, we pass the logits 
          # and take the argmax again in the vanilla decoder.
          self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1]
                             for output in self.outputs[b]]
    else:
      self.outputs, self.losses, self.states = tf.nn.seq2seq.model_with_buckets_states(**model_args)
    if self.seq2seq_mode == 'vae':
      adjust_loss_vae()

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      if opt_algorithm == "sgd":
        logging.info("Using optimizer GradientDescentOptimizer")
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif opt_algorithm == "adagrad":
        print ("Using optimizer AdagradOptimizer")
        lr = 3.0
        init_acc = 0.1
        opt = tf.train.AdagradOptimizer(lr, init_acc)
      elif opt_algorithm == "adadelta":
        print ("Using optimizer AdadeltaOptimizer")  
        rho = 0.95
        epsilon = 1e-6
        opt = tf.train.AdadeltaOptimizer(rho=rho, epsilon=epsilon)        

      #for b in xrange(len(buckets)):
      for loss in self.losses:
        gradients = tf.gradients(loss, params) #self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))
   
    if variable_prefix:
      # save only the variables that belong to the prefix
      logging.info("Using variable prefix={}".format(variable_prefix))
      self.saver = tf.train.Saver({ v.op.name: v for v in tf.global_variables() if v.op.name.startswith(variable_prefix) }, max_to_keep=max_to_keep,
                                  write_version=saver_pb2.SaverDef.V1)
    else:
      self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep,
                                  write_version=saver_pb2.SaverDef.V1)

    if rename_variable_prefix:
      # create a saver that explicitly stores model variables with a prefix
      logging.info("Saving model with new prefix={}".format(rename_variable_prefix))
      self.saver_prefix = tf.train.Saver({v.op.name.replace(variable_prefix, rename_variable_prefix): v for v in tf.global_variables()},
                                         write_version=saver_pb2.SaverDef.V1)

  def get_step_input_feed(self, encoder_inputs, decoder_inputs, target_weights,
                          bucket_id, sequence_length, src_mask, trg_mask,
                          forward_only, hidden=None):
    # Check if the sizes match. Return tuple: input_feed, encoder_size, decoder_size
    encoder_size, decoder_size = self.buckets[-1] if self.single_graph else self.buckets[bucket_id]
#    print("Enc size={} dec size={}".format(encoder_size, decoder_size))
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))
    
    def word_dropout(words):
      if self.seq2seq_mode == 'vae' and self.word_keep_prob < 1.0:
        out = []
        for word in words:
          if word in (data_utils.PAD_ID, data_utils.EOS_ID, data_utils.GO_ID):
            out.append(word)
          else:
            if np.random.uniform() > self.word_keep_prob:
              out.append(data_utils.UNK_ID)
            else:
              out.append(word)
        return out
      else:
        return words
        
    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = word_dropout(decoder_inputs[l])
      if self.scheduled_sample and not forward_only:
        input_feed[self.feed_prev_p.name] = min(1.0, self.global_step.eval() / self.scheduled_sample_steps)
      if self.grammar is not None:
        input_feed[self.grammar.rhs_mask.name] = self.grammar.sampling_rhs
        if not forward_only:
          input_feed[self.grammar.grammar_mask[l].name] = trg_mask[l]
        else:
          input_feed[self.grammar.grammar_full_mask.name] = self.grammar.mask

      input_feed[self.target_weights[l].name] = target_weights[l]
      if l < decoder_size - 1:
        input_feed[self.targets[l].name] = decoder_inputs[l + 1]
    # Since our targets are decoder inputs shifted by one, we need one more.
    input_feed[self.targets[decoder_size - 1].name] = np.zeros(
      [self.batch_size], dtype=np.int32)

    if sequence_length is not None:
      logging.debug("Using sequence length for encoder: feed")
      input_feed[self.sequence_length.name] = sequence_length
    if src_mask is not None:
      logging.debug("Using source mask for decoder: feed")
      input_feed[self.src_mask.name] = src_mask
    if self.bow_mask is not None:
      logging.debug("Using bow mask for decoder: feed")
      input_feed[self.bow_mask.name] = trg_mask
    if hidden is not None:
      logging.debug("Decoding from hidden layer")
      input_feed[self.encoder_states.name] = hidden
    if self.seq2seq_mode == 'vae':
      anneal_scale = 1
      if self.annealing and not forward_only:
        anneal_scale = min(1.0, self.global_step.eval() / self.anneal_steps)
      input_feed[self.anneal_scale.name] = anneal_scale
    return input_feed, encoder_size, decoder_size

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only, sequence_length=None, src_mask=None,
           trg_mask=None):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    input_feed, encoder_size, decoder_size = self.get_step_input_feed(
      encoder_inputs, decoder_inputs, target_weights, 
      bucket_id, sequence_length, src_mask, trg_mask,
      forward_only)
    if self.single_graph:
      bucket_id = -1
    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:                
      # todo: step for this shouldn't really be hardcoded
      if self.seq2seq_mode == 'vae' and self.global_step.eval() % 200 == 0:
        kl_loss, reconstruct_loss = session.run([self.kl_loss[bucket_id],
                                                 self.reconstruct_loss[bucket_id]],
                                                input_feed)
        logging.info("Step {}: KL loss {}, reconstruction loss {} ".format(self.global_step.eval(), 
                                                                           kl_loss,
                                                                           reconstruct_loss))
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
      outputs = session.run(output_feed, input_feed)      
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs   
    else:
      # forward_only true: decoding
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])
      outputs = session.run(output_feed, input_feed)
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs


  def get_state_step(self, session, encoder_inputs, decoder_inputs,
                     target_weights, bucket_id, forward_only,
                     sequence_length=None, src_mask=None, trg_mask=None,
                     hidden=None):
    """Run a step of the model feeding the given inputs, returning state
    
    Args:
    session: tensorflow session to use.
    encoder_inputs: list of numpy int vectors to feed as encoder inputs.
    decoder_inputs: list of numpy int vectors to feed as decoder inputs.
    target_weights: list of numpy float vectors to feed as target weights.
    bucket_id: which bucket of the model to use.
    forward_only: whether to do the backward step or only forward.
    hidden: optional hidden layer to decode from
    Returns:
    A triple consisting of gradient norm (or None if we did not do backward),
    average perplexity, and the outputs.
    
    Raises:
    ValueError: if length of encoder_inputs, decoder_inputs, or
    target_weights disagrees with bucket size for the specified bucket_id.
    """
    input_feed, encoder_size, decoder_size = self.get_step_input_feed(
      encoder_inputs, decoder_inputs, target_weights, 
      bucket_id, sequence_length, src_mask, trg_mask,
      forward_only, hidden=hidden)
    if self.single_graph:
      bucket_id = -1
    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:    
      if self.seq2seq_mode == 'vae' and self.global_step.eval() % 200 == 0:
        kl_loss, reconstruct_loss = session.run([self.kl_loss[bucket_id],
                                                 self.reconstruct_loss[bucket_id]],
                                                input_feed)
        logging.info("Step {}: KL loss {}, reconstruction loss {} ".format(self.global_step.eval(), 
                                                                           kl_loss,
                                                                           reconstruct_loss))
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.

      outputs = session.run(output_feed, input_feed)      
      return outputs[1], outputs[2], None, None  # Gradient norm, loss, no outputs, no states.     
    else:
      # forward_only true: decoding
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      states = [self.states[bucket_id]]
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])
      
      outputs = session.run(output_feed, input_feed)
      output_states = session.run(states, input_feed)
      return None, outputs[0], outputs[1:], output_states  # No gradient norm, loss, outputs, states.

  def get_batch(self, data, bucket_id, encoder="reverse", batch_ptr=None, bookk=None):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.
      batch_ptr: if given, batch is selected using the bucket_id and the idx_map
             and train_offset passed as 'batch'
    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    if batch_ptr is not None:
      train_idx = batch_ptr["offset"]
      idx_map = batch_ptr["idx_map"]
    
    encoder_inputs, decoder_inputs = [], []
    encoder_size, decoder_size = self.buckets[-1] if self.single_graph else self.buckets[bucket_id]

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    enc_input_lengths = []
    for _ in xrange(self.batch_size):
      if batch_ptr is not None:
        if bookk is not None:
          bookk[bucket_id][idx_map[train_idx]] = 1
        encoder_input, decoder_input = data[bucket_id][idx_map[train_idx]]
        train_idx += 1
      else:
        encoder_input, decoder_input = random.choice(data[bucket_id])
      enc_input_lengths.append(len(encoder_input))

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      if encoder == "bidirectional" or encoder == "bow":
        # if we use a bidirectional encoder, inputs are reversed for backward state
        encoder_inputs.append(list(encoder_input + encoder_pad))
      elif encoder == "reverse":
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)       

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights_trg = [], [], []

    src_mask = None
    if self.src_mask is not None:
      src_mask = np.ones((self.batch_size, encoder_size), dtype=np.float32)

    bow_mask = None
    if self.bow_mask is not None:
      bow_mask = np.zeros((self.batch_size, self.target_vocab_size), dtype=np.float32)

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      for batch_idx in xrange(self.batch_size):
        if encoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
          if self.src_mask is not None:
            src_mask[batch_idx, length_idx] = 0
          if self.no_pad_symbol:
            encoder_inputs[batch_idx][length_idx] = data_utils.EOS_ID

        if self.bow_mask is not None:
          word_id = encoder_inputs[batch_idx][length_idx]
          if word_id != data_utils.GO_ID and \
            word_id != data_utils.PAD_ID:
              if self.bow_no_replace:
                bow_mask[batch_idx][word_id] += 1.0
              else:
                bow_mask[batch_idx][word_id] = 1.0
              logging.debug("bow_mask[{}][{}]={}".format(batch_idx, word_id, bow_mask[batch_idx][word_id]))

      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))
             
    grammar_mask = None
    if self.grammar:
      grammar_mask = []
      stack = [[] for _ in range(self.batch_size)]

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      
      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
        if self.no_pad_symbol and decoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
          decoder_inputs[batch_idx][length_idx] = data_utils.EOS_ID

      batch_weights_trg.append(batch_weight)

      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))
      if self.grammar is not None:
        grammar_mask.append(
          self.grammar.stack_step(stack, batch_decoder_inputs[-1]))
    # Make sequence length vector
    sequence_length = None
    if self.sequence_length is not None:
      sequence_length = np.array([enc_input_lengths[batch_idx]
                                for batch_idx in xrange(self.batch_size)], dtype=np.int32)            
    for batch_idx in xrange(self.batch_size):
      logging.debug("encoder input={}".format(encoder_inputs[batch_idx]))
    logging.debug("Sequence length={}".format(sequence_length))
    logging.debug("Source mask={}".format(src_mask))
    if self.bow_mask is not None:
      logging.debug("BOW mask={} (sum={})".format(bow_mask, np.sum(bow_mask)))
      
    trg_mask = None
    if self.bow_mask is not None:
      trg_mask = bow_mask
    elif self.grammar is not None:
      trg_mask = grammar_mask

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights_trg, sequence_length, src_mask, trg_mask
