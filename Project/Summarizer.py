import os
import numpy as np

import tensorflow as tf
from tensorflow.python.layers.core import Dense

import summarizer_model_utils


class Summarizer:

    def __init__(self,
                 word2ind,
                 ind2word,
                 save_path,
                 mode='TRAIN',
                 num_layers_encoder=1,
                 num_layers_decoder=1,
                 embedding_dim=300,
                 rnn_size_encoder=256,
                 rnn_size_decoder=256,
                 learning_rate=0.001,
                 learning_rate_decay=0.9,
                 learning_rate_decay_steps=100,
                 max_lr=0.01,
                 keep_probability=0.8,
                 batch_size=64,
                 beam_width=10,
                 epochs=20,
                 eos="<EOS>",
                 sos="<SOS>",
                 pad='<PAD>',
                 clip=5,
                 inference_targets=False,
                 pretrained_embeddings_path=None,
                 summary_dir=None,
                 use_cyclic_lr=False):
        """

        Args:
            word2ind: lookup dict from word to index.
            ind2word: lookup dict from index to word.
            save_path: path to save the tf model to in the end.
            mode: String. 'TRAIN' or 'INFER'. depending on which mode we use
                  a different graph is created.
            num_layers_encoder: Float. Number of encoder layers. defaults to 1.
            num_layers_decoder: Float. Number of decoder layers. defaults to 1.
            embedding_dim: dimension of the embedding vectors in the embedding matrix.
                           every word has a embedding_dim 'long' vector.
            rnn_size_encoder: Integer. number of hidden units in encoder. defaults to 256.
            rnn_size_decoder: Integer. number of hidden units in decoder. defaults to 256.
            learning_rate: Float.
            learning_rate_decay: only if exponential learning rate is used.
            learning_rate_decay_steps: Integer.
            max_lr: only used if cyclic learning rate is used.
            keep_probability: Float.
            batch_size: Integer. Size of minibatches.
            beam_width: Integer. Only used in inference, for Beam Search.('INFER'-mode)
            epochs: Integer. Number of times the training is conducted
                    on the whole training data.
            eos: EndOfSentence tag.
            sos: StartOfSentence tag.
            pad: Padding tag.
            clip: Value to clip the gradients to in training process.
            inference_targets:
            pretrained_embeddings_path: Path to pretrained embeddings. Has to be .npy
            summary_dir: Directory the summaries are written to for tensorboard.
            use_cyclic_lr: Boolean.
        """

        self.word2ind = word2ind
        self.ind2word = ind2word
        self.vocab_size = len(word2ind)
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.rnn_size_encoder = rnn_size_encoder
        self.rnn_size_decoder = rnn_size_decoder
        self.save_path = save_path
        self.embedding_dim = embedding_dim
        self.mode = mode.upper()
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.keep_probability = keep_probability
        self.batch_size = batch_size
        self.beam_width = beam_width
        self.eos = eos
        self.sos = sos
        self.clip = clip
        self.pad = pad
        self.epochs = epochs
        self.inference_targets = inference_targets
        self.pretrained_embeddings_path = pretrained_embeddings_path
        self.use_cyclic_lr = use_cyclic_lr
        self.max_lr = max_lr
        self.summary_dir = summary_dir

    def build_graph(self):
        self.add_placeholders()
        self.add_embeddings()
        self.add_lookup_ops()
        self.initialize_session()
        self.add_seq2seq()
        self.saver = tf.train.Saver()
        print('Graph built.')

    def add_placeholders(self):
        self.ids_1 = tf.placeholder(tf.int32,
                                    shape=[None, None],
                                    name='ids_source')
        self.ids_2 = tf.placeholder(tf.int32,
                                    shape=[None, None],
                                    name='ids_target')
        self.sequence_lengths_1 = tf.placeholder(tf.int32,
                                                 shape=[None],
                                                 name='sequence_length_source')
        self.sequence_lengths_2 = tf.placeholder(tf.int32,
                                                 shape=[None],
                                                 name='sequence_length_target')
        self.maximum_iterations = tf.reduce_max(self.sequence_lengths_2,
                                                name='max_dec_len')

    def create_word_embedding(self, embed_name, vocab_size, embed_dim):
        """Creates embedding matrix in given shape - [vocab_size, embed_dim].
        """
        embedding = tf.get_variable(embed_name,
                                    shape=[vocab_size, embed_dim],
                                    dtype=tf.float32)
        return embedding

    def add_embeddings(self):
        """Creates the embedding matrix. In case path to pretrained embeddings is given,
           that embedding is loaded. Otherwise created.
        """
        if self.pretrained_embeddings_path is not None:
            self.embedding = tf.Variable(np.load(self.pretrained_embeddings_path),
                                         name='embedding')
            print('Loaded pretrained embeddings.')
        else:
            self.embedding = self.create_word_embedding('embedding',
                                                        self.vocab_size,
                                                        self.embedding_dim)

    def add_lookup_ops(self):
        """Additional lookup operation for both source embedding and target embedding matrix.
        """
        self.word_embeddings_1 = tf.nn.embedding_lookup(self.embedding,
                                                        self.ids_1,
                                                        name='word_embeddings_1')
        self.word_embeddings_2 = tf.nn.embedding_lookup(self.embedding,
                                                        self.ids_2,
                                                        name='word_embeddings_2')

    def make_rnn_cell(self, rnn_size, keep_probability):
        """Creates LSTM cell wrapped with dropout.
        """
        cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_probability)
        return cell

    def make_attention_cell(self, dec_cell, rnn_size, enc_output, lengths, alignment_history=False):
        """Wraps the given cell with Bahdanau Attention.
        """
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,
                                                                   memory=enc_output,
                                                                   memory_sequence_length=lengths,
                                                                   name='BahdanauAttention')

        return tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell,
                                                   attention_mechanism=attention_mechanism,
                                                   attention_layer_size=None,
                                                   output_attention=False,
                                                   alignment_history=alignment_history)

    def triangular_lr(self, current_step):
        """cyclic learning rate - exponential range."""
        step_size = self.learning_rate_decay_steps
        base_lr = self.learning_rate
        max_lr = self.max_lr

        cycle = tf.floor(1 + current_step / (2 * step_size))
        x = tf.abs(current_step / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * tf.maximum(0.0, tf.cast((1.0 - x), dtype=tf.float32)) * (0.99999 ** tf.cast(
            current_step,
            dtype=tf.float32))
        return lr


    def add_seq2seq(self):
        """Creates the sequence to sequence architecture."""
        with tf.variable_scope('dynamic_seq2seq', dtype=tf.float32):
            # Encoder
            encoder_outputs, encoder_state = self.build_encoder()

            # Decoder
            logits, sample_id, final_context_state = self.build_decoder(encoder_outputs,
                                                                        encoder_state)
            if self.mode == 'TRAIN':

                # Loss
                loss = self.compute_loss(logits)
                self.train_loss = loss
                self.eval_loss = loss
                self.global_step = tf.Variable(0, trainable=False)


                # cyclic learning rate
                if self.use_cyclic_lr:
                    self.learning_rate = self.triangular_lr(self.global_step)

                # exponential learning rate
                else:
                    self.learning_rate = tf.train.exponential_decay(
                        self.learning_rate,
                        self.global_step,
                        decay_steps=self.learning_rate_decay_steps,
                        decay_rate=self.learning_rate_decay,
                        staircase=True)

                # Optimizer
                opt = tf.train.AdamOptimizer(self.learning_rate)


                # Gradients
                if self.clip > 0:
                    grads, vs = zip(*opt.compute_gradients(self.train_loss))
                    grads, _ = tf.clip_by_global_norm(grads, self.clip)
                    self.train_op = opt.apply_gradients(zip(grads, vs),
                                                        global_step=self.global_step)
                else:
                    self.train_op = opt.minimize(self.train_loss,
                                                 global_step=self.global_step)



            elif self.mode == 'INFER':
                loss = None
                self.infer_logits, _, self.final_context_state, self.sample_id = logits, loss, final_context_state, sample_id
                self.sample_words = self.sample_id

    def build_encoder(self):
        """The encoder. Bidirectional LSTM."""

        with tf.variable_scope("encoder"):
            fw_cell = self.make_rnn_cell(self.rnn_size_encoder // 2, self.keep_probability)
            bw_cell = self.make_rnn_cell(self.rnn_size_encoder // 2, self.keep_probability)

            for _ in range(self.num_layers_encoder):
                (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cell,
                    cell_bw=bw_cell,
                    inputs=self.word_embeddings_1,
                    sequence_length=self.sequence_lengths_1,
                    dtype=tf.float32)
                encoder_outputs = tf.concat((out_fw, out_bw), -1)

            bi_state_c = tf.concat((state_fw.c, state_bw.c), -1)
            bi_state_h = tf.concat((state_fw.h, state_bw.h), -1)
            bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
            encoder_state = tuple([bi_lstm_state] * self.num_layers_encoder)

            return encoder_outputs, encoder_state


    def build_decoder(self, encoder_outputs, encoder_state):

        sos_id_2 = tf.cast(self.word2ind[self.sos], tf.int32)
        eos_id_2 = tf.cast(self.word2ind[self.eos], tf.int32)
        self.output_layer = Dense(self.vocab_size, name='output_projection')

        # Decoder.
        with tf.variable_scope("decoder") as decoder_scope:

            cell, decoder_initial_state = self.build_decoder_cell(
                encoder_outputs,
                encoder_state,
                self.sequence_lengths_1)

            # Train
            if self.mode != 'INFER':

                helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs=self.word_embeddings_2,
                    sequence_length=self.sequence_lengths_2,
                    embedding=self.embedding,
                    sampling_probability=0.5,
                    time_major=False)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             helper,
                                                             decoder_initial_state,
                                                             output_layer=self.output_layer)

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    output_time_major=False,
                    maximum_iterations=self.maximum_iterations,
                    swap_memory=False,
                    impute_finished=True,
                    scope=decoder_scope
                )

                sample_id = outputs.sample_id
                logits = outputs.rnn_output


            # Inference
            else:
                start_tokens = tf.fill([self.batch_size], sos_id_2)
                end_token = eos_id_2

                # beam search
                if self.beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embedding,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=self.output_layer,
                    )

                # greedy
                else:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding,
                                                                      start_tokens,
                                                                      end_token)

                    my_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                                 helper,
                                                                 decoder_initial_state,
                                                                 output_layer=self.output_layer)
                if self.inference_targets:
                    maximum_iterations = self.maximum_iterations
                else:
                    maximum_iterations = None

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    maximum_iterations=maximum_iterations,
                    output_time_major=False,
                    impute_finished=False,
                    swap_memory=False,
                    scope=decoder_scope)

                if self.beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    def build_decoder_cell(self, encoder_outputs, encoder_state,
                           sequence_lengths_1):
        """Builds the attention decoder cell. If mode is inference performs tiling
           Passes last encoder state.
        """

        memory = encoder_outputs

        if self.mode == 'INFER' and self.beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(memory,
                                                   multiplier=self.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state,
                                                          multiplier=self.beam_width)
            sequence_lengths_1 = tf.contrib.seq2seq.tile_batch(sequence_lengths_1,
                                                               multiplier=self.beam_width)
            batch_size = self.batch_size * self.beam_width

        else:
            batch_size = self.batch_size

        # MY APPROACH
        if self.num_layers_decoder is not None:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.make_rnn_cell(self.rnn_size_decoder, self.keep_probability) for _ in
                 range(self.num_layers_decoder)])

        else:
            lstm_cell = self.make_rnn_cell(self.rnn_size_decoder, self.keep_probability)

        # attention cell
        cell = self.make_attention_cell(lstm_cell,
                                        self.rnn_size_decoder,
                                        memory,
                                        sequence_lengths_1)

        decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

        return cell, decoder_initial_state


    def compute_loss(self, logits):
        """Compute the loss during optimization."""
        target_output = self.ids_2
        max_time = self.maximum_iterations

        target_weights = tf.sequence_mask(self.sequence_lengths_2,
                                          max_time,
                                          dtype=tf.float32,
                                          name='mask')

        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=target_output,
                                                weights=target_weights,
                                                average_across_timesteps=True,
                                                average_across_batch=True, )
        return loss


    def train(self,
              inputs,
              targets,
              restore_path=None,
              validation_inputs=None,
              validation_targets=None):
        """Performs the training process. Runs training step in every epoch.
           Shuffles input data before every epoch.
           Optionally: - add tensorboard summaries.
                       - restoring previous model and retraining on top.
                       - evaluation step.
        """
        assert len(inputs) == len(targets)

        if self.summary_dir is not None:
            self.add_summary()

        self.initialize_session()
        if restore_path is not None:
            self.restore_session(restore_path)

        best_score = np.inf
        nepoch_no_imprv = 0

        inputs = np.array(inputs)
        targets = np.array(targets)

        for epoch in range(self.epochs + 1):
            print('-------------------- Epoch {} of {} --------------------'.format(epoch,
                                                                                    self.epochs))

            # shuffle the input data before every epoch.
            shuffle_indices = np.random.permutation(len(inputs))
            inputs = inputs[shuffle_indices]
            targets = targets[shuffle_indices]

            # run training epoch
            score = self.run_epoch(inputs, targets, epoch)

            # evaluate model
            if validation_inputs is not None and validation_targets is not None:
                self.run_evaluate(validation_inputs, validation_targets, epoch)


            if score <= best_score:
                nepoch_no_imprv = 0
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                self.saver.save(self.sess, self.save_path)
                best_score = score
                print("--- new best score ---\n\n")
            else:
                # warm up epochs for the model
                if epoch > 10:
                    nepoch_no_imprv += 1
                # early stopping
                if nepoch_no_imprv >= 5:
                    print("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                    break

    def infer(self, inputs, restore_path, targets=None):
        """Runs inference process. No training takes place.
           Returns the predicted ids for every sentence.
        """
        self.initialize_session()
        self.restore_session(restore_path)

        prediction_ids = []
        if targets is not None:
            feed, _, sequence_lengths_2 = self.get_feed_dict(inputs, trgts=targets)
        else:
            feed, _ = self.get_feed_dict(inputs)

        infer_logits, s_ids = self.sess.run([self.infer_logits, self.sample_words], feed_dict=feed)
        prediction_ids.append(s_ids)

        # for (inps, trgts) in summarizer_model_utils.minibatches(inputs, targets, self.batch_size):
        #     feed, _, sequence_lengths= self.get_feed_dict(inps, trgts=trgts)
        #     infer_logits, s_ids = self.sess.run([self.infer_logits, self.sample_words], feed_dict = feed)
        #     prediction_ids.append(s_ids)

        return prediction_ids

    def run_epoch(self, inputs, targets, epoch):
        """Runs a single epoch.
           Returns the average loss value on the epoch."""
        batch_size = self.batch_size
        nbatches = (len(inputs) + batch_size - 1) // batch_size
        losses = []

        for i, (inps, trgts) in enumerate(summarizer_model_utils.minibatches(inputs,
                                                                             targets,
                                                                             batch_size)):
            if inps is not None and trgts is not None:
                fd, sl, s2 = self.get_feed_dict(inps,
                                                trgts=trgts)

                if i % 10 == 0 and self.summary_dir is not None:
                    _, train_loss, training_summ = self.sess.run([self.train_op,
                                                                  self.train_loss,
                                                                  self.training_summary],
                                                                 feed_dict=fd)
                    self.training_writer.add_summary(training_summ, epoch*nbatches + i)

                else:
                    _, train_loss = self.sess.run([self.train_op, self.train_loss],
                                                  feed_dict=fd)

                if i % 2 == 0 or i == (nbatches - 1):
                    print('Iteration: {} of {}\ttrain_loss: {:.4f}'.format(i, nbatches - 1, train_loss))
                losses.append(train_loss)

            else:
                print('Minibatch empty.')
                continue

        avg_loss = self.sess.run(tf.reduce_mean(losses))
        print('Average Score for this Epoch: {}'.format(avg_loss))

        return avg_loss

    def run_evaluate(self, inputs, targets, epoch):
        """Runs evaluation on validation inputs and targets.
        Optionally: - writes summary to Tensorboard.
        """
        if self.summary_dir is not None:
            eval_losses = []
            for inps, trgts in summarizer_model_utils.minibatches(inputs, targets, self.batch_size):
                fd, sl, s2 = self.get_feed_dict(inps, trgts)
                eval_loss = self.sess.run([self.eval_loss], feed_dict=fd)
                eval_losses.append(eval_loss)

            avg_eval_loss = self.sess.run(tf.reduce_mean(eval_losses))

            print('Eval_loss: {}\n'.format(avg_eval_loss))
            eval_summ = self.sess.run([self.eval_summary], feed_dict=fd)
            self.eval_writer.add_summary(eval_summ, epoch)

        else:
            eval_losses = []
            for inps, trgts in summarizer_model_utils.minibatches(inputs, targets, self.batch_size):
                fd, sl, s2 = self.get_feed_dict(inps, trgts)
                eval_loss = self.sess.run([self.eval_loss], feed_dict=fd)
                eval_losses.append(eval_loss)

            avg_eval_loss = self.sess.run(tf.reduce_mean(eval_losses))

            print('Eval_loss: {}\n'.format(avg_eval_loss))



    def get_feed_dict(self, inps, trgts=None):
        """Creates the feed_dict that is fed into training or inference network.
           Pads inputs and targets.
           Returns feed_dict and sequence_length(s) depending on training mode.
        """
        if self.mode != 'INFER':
            inp_ids, sequence_lengths_1 = summarizer_model_utils.pad_sequences(inps,
                                                                               self.word2ind[self.pad],
                                                                               tail=False)

            feed = {
                self.ids_1: inp_ids,
                self.sequence_lengths_1: sequence_lengths_1
            }

            if trgts is not None:
                trgt_ids, sequence_lengths_2 = summarizer_model_utils.pad_sequences(trgts,
                                                                                    self.word2ind[self.pad],
                                                                                    tail=True)
                feed[self.ids_2] = trgt_ids
                feed[self.sequence_lengths_2] = sequence_lengths_2

                return feed, sequence_lengths_1, sequence_lengths_2

        else:

            inp_ids, sequence_lengths_1 = summarizer_model_utils.pad_sequences(inps,
                                                                               self.word2ind[self.pad],
                                                                               tail=False)

            feed = {
                self.ids_1: inp_ids,
                self.sequence_lengths_1: sequence_lengths_1
            }

            if trgts is not None:
                trgt_ids, sequence_lengths_2 = summarizer_model_utils.pad_sequences(trgts,
                                                                                    self.word2ind[self.pad],
                                                                                    tail=True)

                feed[self.sequence_lengths_2] = sequence_lengths_2

                return feed, sequence_lengths_1, sequence_lengths_2
            else:
                return feed, sequence_lengths_1

    def initialize_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def restore_session(self, restore_path):
        self.saver.restore(self.sess, restore_path)
        print('Done.')

    def add_summary(self):
        """Summaries for Tensorboard."""
        self.training_summary = tf.summary.scalar('training_loss', self.train_loss)
        self.eval_summary = tf.summary.scalar('evaluation_loss', self.eval_loss)
        self.training_writer = tf.summary.FileWriter(self.summary_dir,
                                                     tf.get_default_graph())
        self.eval_writer = tf.summary.FileWriter(self.summary_dir)
