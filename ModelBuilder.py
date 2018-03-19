
import tensorflow as tf

class ModelBuilder:

    def __init__(self, max_caption_len, n_lstm_units, batch_size, vocab_size):

        self.channels=4032
        self.window_size=121
        self.dim_embed=512
        self.n_lstm_units = n_lstm_units
        self.max_caption_len = max_caption_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.lstm_drop_rate = 0.5
        self.attention_loss_factor= 0.01
        self.fc_kernel_initializer = tf.random_uniform_initializer(minval= -0.08,
                                                                    maxval= 0.08)
        self.fc_kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)

    def build_train_graph(self, conv_feats, sentences, masks):

        # placeholders...
        # conv_feats = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_ctx, self.dim_ctx])
        # sentences = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_caption_len])
        # masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_caption_len])
        word_embedding = tf.get_variable(shape=[self.vocab_size, self.dim_embed],
                                         initializer=self.fc_kernel_initializer,
                                         regularizer=self.fc_kernel_regularizer,
                                         name='word_embed')
        # Setup the LSTM
        lstm = tf.nn.rnn_cell.LSTMCell(
            self.n_lstm_units,
            initializer=self.fc_kernel_initializer)

        lstm = tf.nn.rnn_cell.DropoutWrapper(
            lstm,
            input_keep_prob=1.0 - self.lstm_drop_rate,
            output_keep_prob=1.0 - self.lstm_drop_rate,
            state_keep_prob=1.0 - self.lstm_drop_rate)

        # Initialize the LSTM using the mean conv_feats
        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(conv_feats, axis=1)
            initial_cell_memory, initial_hidden_state = self.initialize(context_mean, is_train=True)

        masks=tf.cast(masks, dtype=tf.float32)

        alphas = []
        cross_entropies = []

        last_hidden_state = initial_hidden_state
        last_cell_memory = initial_cell_memory
        last_word = tf.ones([self.batch_size], tf.int32)
        last_state_tuple = last_cell_memory, last_hidden_state

        # Generate the words one by one
        for idx in range(self.max_caption_len-2):
            # Attention mechanism
            with tf.variable_scope("attend"):
                alpha = self.attend(conv_feats, last_hidden_state, is_train=True)
                temp_Z = conv_feats * tf.expand_dims(alpha, 2)
                Z = tf.reduce_sum(temp_Z, axis=1)

                tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1), [1, self.window_size])
                masked_alpha = alpha * tiled_masks
                alphas.append(tf.reshape(masked_alpha, [-1]))

            # Find embedding for the last word
            with tf.variable_scope("word_embedding") and tf.device("/cpu:0"):

                word_embed = tf.nn.embedding_lookup(word_embedding, last_word)

            # Apply the LSTM
            with tf.variable_scope("lstm"):
                current_input = tf.concat([Z, word_embed], 1)
                output, current_state_tuple = lstm(current_input, last_state_tuple)

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             Z,
                                             word_embed],
                                            axis=1)

                expanded_output = tf.layers.dropout(inputs=expanded_output, training=True)
                # use 1 fc layer to map to vacab_size
                logits = tf.layers.dense(inputs=expanded_output, units=self.vocab_size, trainable=True, name='decode_fc')
                # [b_size, vocab_size]

            # Compute the loss for this step
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=sentences[:, idx+1],
                logits=logits)
            masked_cross_entropy = cross_entropy * masks[:, idx]
            cross_entropies.append(masked_cross_entropy)

            last_hidden_state = output
            last_state_tuple = current_state_tuple
            last_word = sentences[:, idx+1]
            tf.get_variable_scope().reuse_variables()
            # End for

        # Compute final loss...
        cross_entropies = tf.stack(cross_entropies, axis=1)
        cross_entropy_loss = tf.reduce_sum(cross_entropies) / tf.reduce_sum(masks)

        alphas = tf.stack(alphas, axis=1)
        alphas = tf.reshape(alphas, [self.batch_size, self.window_size, -1])
        attentions = tf.reduce_sum(alphas, axis=2)
        diffs = tf.ones_like(attentions) - attentions
        attention_loss =  self.attention_loss_factor * tf.nn.l2_loss(diffs)/(self.batch_size * self.window_size)

        reg_loss = tf.losses.get_regularization_loss()

        total_loss = cross_entropy_loss + attention_loss + reg_loss

        return total_loss #Total loss of a batch

    def build_test_graph(self, conv_feats):

        print("Building test model...")

        word_embedding = tf.get_variable(shape=[self.vocab_size, self.dim_embed],
                                         initializer=self.fc_kernel_initializer,
                                         regularizer=self.fc_kernel_regularizer,
                                         trainable=False,
                                         name='word_embed')

        lstm = tf.nn.rnn_cell.LSTMCell(
            self.n_lstm_units,
            initializer=self.fc_kernel_initializer)

        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(conv_feats, axis=1)
            initial_memory, initial_output = self.initialize(context_mean, is_train=False)
            initial_state = initial_memory, initial_output

        last_hidden_state = initial_output
        last_word = tf.ones([self.batch_size], tf.int32)
        last_state_tuple = initial_state
        predictions = []
        max_caption_len=self.max_caption_len

        for idx in range(max_caption_len-2):
            # Attention mechanism
            with tf.variable_scope("attend"):
                alpha = self.attend(conv_feats, last_hidden_state, is_train=False)
                temp_Z = conv_feats * tf.expand_dims(alpha, 2)
                Z = tf.reduce_sum(temp_Z, axis=1)

            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(word_embedding,
                                                    last_word)
                # Apply the LSTM
            with tf.variable_scope("lstm"):
                current_input = tf.concat([Z, word_embed], 1)
                output, current_state_tuple = lstm(current_input, last_state_tuple)

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             Z,
                                             word_embed],
                                            axis=1)
                expanded_output = tf.layers.dropout(inputs=expanded_output, training=False)
                # use 1 fc layer to decode
                logits = tf.layers.dense(inputs=expanded_output, units=self.vocab_size, trainable=False, name='decode_fc')

                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)

            tf.get_variable_scope().reuse_variables()

            last_hidden_state = output
            last_word = prediction
            last_state_tuple = current_state_tuple

        print("Test graph built.")

        captions = tf.transpose(predictions)
        return captions


    def attend(self, conv_feats, last_hidden_state, is_train):
        #Attention...
        reshaped_conv_feats = tf.reshape(conv_feats, [-1, self.channels])
        reshaped_conv_feats = tf.layers.dropout(reshaped_conv_feats, training=is_train)
        last_hidden_state = tf.layers.dropout(last_hidden_state, training=is_train)

        if(is_train):
            logits1 = tf.layers.dense(
                inputs=reshaped_conv_feats,
                units=1,
                use_bias=False,
                trainable=True,
                kernel_initializer=self.fc_kernel_initializer,
                kernel_regularizer=self.fc_kernel_regularizer,
                name='attend_fc1')
        else:
            logits1 = tf.layers.dense(
                inputs=reshaped_conv_feats,
                units=1,
                use_bias=False,
                trainable=False,
                kernel_initializer=self.fc_kernel_initializer,
                kernel_regularizer=None,
                name='attend_fc1')

        logits1 = tf.reshape(logits1, [-1, self.window_size])  # [b_size, 121]

        if(is_train):
            logits2 = tf.layers.dense(
                inputs=last_hidden_state,
                units=1,
                use_bias=False,
                trainable=True,
                kernel_initializer=self.fc_kernel_initializer,
                kernel_regularizer=self.fc_kernel_regularizer,
                name='attend_fc2')
        else:
            logits2 = tf.layers.dense(
                inputs=last_hidden_state,
                units=1,
                use_bias=False,
                trainable=False,
                kernel_initializer=self.fc_kernel_initializer,
                kernel_regularizer=None,
                name='attend_fc2')
        # [b_size, 121]

        logits = logits1 + logits2
        alpha = tf.nn.softmax(logits)

        return alpha

    def initialize(self, context_mean, is_train):
        """ Initialize the LSTM using the mean context. """
        context_mean = tf.layers.dropout(context_mean, training=is_train)
        initial_cell_memory = tf.layers.dense(context_mean, units=self.n_lstm_units, trainable=is_train, name='init_mem_fc')
        initial_hidden_state = tf.layers.dense(context_mean, units=self.n_lstm_units, trainable=is_train, name='init_out_fc')

        return initial_cell_memory, initial_hidden_state
