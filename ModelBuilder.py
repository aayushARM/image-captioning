
import tensorflow as tf

class ModelBuilder:

    def __init__(self, max_caption_len, n_lstm_units, batch_size, vocab_size):

        self.channels=4032 #channel width of conv_feats
        self.window_size=121 # 11 x 11
        self.dim_embed=512 #size of word embeddings
        self.n_lstm_units = n_lstm_units
        self.max_caption_len = max_caption_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.lstm_keep_prob = 0.5
        self.attention_loss_contrib = 0.01 # determines how much will attention_loss contribute to total loss
        
        #create initializer and regularizer to be used in all the dense layers
        self.fc_kernel_initializer = tf.random_uniform_initializer(minval= -0.08,
                                                                    maxval= 0.08)
        self.fc_kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)

    def build_train_graph(self, conv_feats, sentences, masks):

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
            input_keep_prob = self.lstm_keep_prob,
            output_keep_prob = self.lstm_keep_prob,
            state_keep_prob = self.lstm_keep_prob)

        # NOTE: Everything below is in accordance with the paper.
        # tf.variable_scope() groups variables into a common scope, useful when visualizing graph using Tensorboard.
        # First, initialize LSTM using the mean of conv_feats...
        with tf.variable_scope("initialize"):
            conv_feat_mean = tf.reduce_mean(conv_feats, axis=1)
            initial_cell_memory, initial_hidden_state = self.initialize(conv_feat_mean, is_train=True)

        masks=tf.cast(masks, dtype=tf.float32)
        #inputs for 1st step of LSTM
        last_hidden_state = initial_hidden_state
        last_cell_memory = initial_cell_memory
        last_word = tf.ones([self.batch_size], tf.int32) # because index of start-token = 1
        last_state_tuple = last_cell_memory, last_hidden_state

        alphas = []
        cross_entropies = []

        for idx in range(self.max_caption_len-2):
            # Attention mechanism
            with tf.variable_scope("attend"):
                alpha = self.attend(conv_feats, last_hidden_state, is_train=True)
                temp_Z = conv_feats * tf.expand_dims(alpha, 2)
                Z = tf.reduce_sum(temp_Z, axis=1) #final Soft Attention "context vector", to be fed into LSTM

                tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1), [1, self.window_size])
                #alpha needs to be masked before calculating attention loss, just the way we mask cross-entropy loss
                masked_alpha = alpha * tiled_masks
                alphas.append(tf.reshape(masked_alpha, [-1]))

            # Find embedding for the last word, usually faster on CPU than GPU
            with tf.variable_scope("word_embedding") and tf.device("/cpu:0"):
                word_embed = tf.nn.embedding_lookup(word_embedding, last_word)

            # Run LSTM step
            with tf.variable_scope("lstm"):
                #because LSTMCell only has two argumnets, the 3rd input(Z) has to be concatenated with word_embed.
                current_input = tf.concat([Z, word_embed], 1)
                output, current_state_tuple = lstm(current_input, last_state_tuple)

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             Z,
                                             word_embed],
                                            axis=1)

                expanded_output = tf.layers.dropout(inputs=expanded_output, training=True)
                # use 1 fc layer to map to from dim_embed to vocab_size
                logits = tf.layers.dense(inputs=expanded_output, units=self.vocab_size, trainable=True, name='decode_fc')
                # Shape: [b_size, vocab_size]

            # Compute the cross-entropy loss for this step with naive word-to-word comparison
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=sentences[:, idx+1],
                logits=logits)
            # mask it so that losses of steps after end_token(</S>) is fed are not considered
            masked_cross_entropy = cross_entropy * masks[:, idx]
            cross_entropies.append(masked_cross_entropy)

            #prepare for next step
            last_hidden_state = output
            last_state_tuple = current_state_tuple
            last_word = sentences[:, idx+1]
            tf.get_variable_scope().reuse_variables()
            # End for

        # Collect all cross-entropies, add them, and find average loss/word
        cross_entropies = tf.stack(cross_entropies, axis=1)
        cross_entropy_loss = tf.reduce_sum(cross_entropies) / tf.reduce_sum(masks) #tf.reduce_sum(masks) will return
                                                                                   #total no. of words in entire batch

        # Collect alphas
        alphas = tf.stack(alphas, axis=1)
        alphas = tf.reshape(alphas, [self.batch_size, self.window_size, -1]) # -1 = determine automatically based
                                                                             # on other 2 dims
        # attentions = summation of all channels(the last dimension above) of alphas of all batches
        attentions = tf.reduce_sum(alphas, axis=2) #now, Shape: [b_size, window_size]

        diffs = tf.ones_like(attentions) - attentions
        # l2_loss() will try to minimize "squared differences" between the
        # estimated values (attentions) and target values (tf.ones_like(attentions))
        attention_loss = self.attention_loss_contrib * tf.nn.l2_loss(diffs) / (self.batch_size * self.window_size)
        # divide by (b_size * window_size), since we want average, and shape of 'attentions', and hence
        # 'diffs' tensor, is [b_size, window_size]

        # TF automatically maintains losses of all regularizers declared using tf.contrib.layers.l2_regularizer()
        regularization_loss = tf.losses.get_regularization_loss()

        total_loss = cross_entropy_loss + attention_loss + regularization_loss

        return total_loss

    # For inference
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

            with tf.variable_scope("attend"):
                alpha = self.attend(conv_feats, last_hidden_state, is_train=False)
                temp_Z = conv_feats * tf.expand_dims(alpha, 2)
                Z = tf.reduce_sum(temp_Z, axis=1)

            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(word_embedding,
                                                    last_word)

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
            # prepare for next step
            last_hidden_state = output
            last_word = prediction
            last_state_tuple = current_state_tuple

        print("Test graph built.")

        captions = tf.transpose(predictions)
        return captions

    def attend(self, conv_feats, last_hidden_state, is_train):

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
        # Need to apply softmax, as descibed in paper. Output is alpha
        alpha = tf.nn.softmax(logits)

        return alpha

    def initialize(self, conv_feat_mean, is_train):
        #Initialize the LSTM using the mean of conv_feats.
        conv_feat_mean = tf.layers.dropout(conv_feat_mean, training=is_train)
        initial_cell_memory = tf.layers.dense(conv_feat_mean, units=self.n_lstm_units, trainable=is_train, name='init_mem_fc')
        initial_hidden_state = tf.layers.dense(conv_feat_mean, units=self.n_lstm_units, trainable=is_train, name='init_out_fc')

        return initial_cell_memory, initial_hidden_state
