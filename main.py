
from .ModelBuilder import ModelBuilder
from .utils import *
import numpy as np
import time

gcloud_bucket = 'gs://image-captioning-196706-bucket/'
nasnet_path = gcloud_bucket + 'MSCOCO_Preprocessed/frozen_nasnet-large.pb'
save_path = gcloud_bucket + 'saved_checkpoints/'
# nasnet_path = '/media/aayush/01D0B5E0A4CEC360/ML and FY Project stuff/Projects/FY Project/MSCOCO_Attention_NASNet/data/frozen_nasnet-large.pb'
# save_path = '/media/aayush/01D0B5E0A4CEC360/ML and FY Project stuff/Projects/FY Project/MSCOCO_Attention_NASNet/saved_checkpoints'
reader = tf.TFRecordReader()
min_queue_examples = 4600
num_reader_threads = 3
num_preprocess_threads = 6
max_caption_len = 23
vocab_size = 10204  # 10329
n_lstm_units = 512

with tf.gfile.Open(nasnet_path, mode='rb') as f:
    fileContent = f.read()
    nasnet_graph_def = tf.GraphDef()
    nasnet_graph_def.ParseFromString(fileContent)

def train():
    #Declare necessary constants
    file_names = tf.gfile.Glob(gcloud_bucket + 'MSCOCO_Preprocessed/train*')
    #file_names = glob.glob('/media/aayush/Local Disk/' + 'MSCOCO_Preprocessed/train*')
    batch_size = 32
    initial_learning_rate = 0.00001
    learning_rate=tf.constant(initial_learning_rate)
    total_pairs = 587605 #591435
    num_epochs_per_decay = 0.7
    clip_gradients = 5.0
    num_batches_per_epoch = int(np.ceil(total_pairs / batch_size))
    num_epochs = 100000
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

    print('Building input pipeline...\n')
    images, sentences, masks = build_inputs(file_names, reader, min_queue_examples, num_reader_threads,
                                                  num_preprocess_threads, batch_size)

    # Add remaining necessary padding
    caption_len = tf.shape(masks)[1]
    paddings = [[0, 0], [0, 23-caption_len]]
    sentences = tf.pad(sentences, paddings, "CONSTANT")
    masks = tf.pad(masks, paddings, "CONSTANT")

    print('Building Nasnet inference graph...\n')
    tf.import_graph_def(nasnet_graph_def, input_map={"input": images})
    graph = tf.get_default_graph()
    conv_feats = graph.get_tensor_by_name("import/final_layer/Relu:0")
    conv_feats = tf.reshape(conv_feats, [batch_size, 121, 4032])

    print('Building training LSTM graph...\n')
    lstm_model = ModelBuilder(max_caption_len, n_lstm_units, batch_size, vocab_size)
    batch_loss= lstm_model.build_train_graph(conv_feats, sentences, masks)

    global_step = tf.Variable(83625, trainable=False)

    def decay_function(learning_rate, global_step):
        return tf.train.exponential_decay(learning_rate, global_step,
                                           decay_steps, 0.90, staircase=True)

    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        adam_optimizer = tf.train.AdamOptimizer(initial_learning_rate)

        train_op = tf.contrib.layers.optimize_loss(
            loss=batch_loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=adam_optimizer,
            clip_gradients=clip_gradients,
            learning_rate_decay_fn=decay_function)

    sess=tf.InteractiveSession()
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3)
    saver.restore(sess, tf.train.latest_checkpoint(save_path))
    for epoch_no in range(num_epochs):
        for batch_no in range(num_batches_per_epoch):
           start_time = time.time()
           _, curr_batch_loss = sess.run([train_op, batch_loss])

           print("Current Cost: ", curr_batch_loss, "\t Epoch {}/{}".format(epoch_no, num_epochs),
              "\t Batch {}/{} | {:.2f}s".format(batch_no, num_batches_per_epoch, time.time() - start_time))

           if ((batch_no % 1000) == 0) and (batch_no >0):
               print("Saving model...")
               saver.save(sess, save_path + 'model', global_step=global_step)

        print("Saving the model from epoch: ", epoch_no)
        saver.save(sess, save_path + 'model', global_step=global_step)

def test_raw():

    file_names = tf.gfile.Glob('/media/aayush/01D0B5E0A34BF3F0/Processed MSCOCO Images/train2017/000000000656.jpg')
    num_images = len(file_names)

    filename_queue = tf.train.string_input_producer(file_names, shuffle=False)

    reader = tf.WholeFileReader()
    image_name, value = reader.read(filename_queue)
    #image_name = image_name[62:]
    image = tf.image.decode_jpeg(value, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    enqueue_list=[]
    enqueue_list.append([image])
    images = tf.train.batch_join(
        enqueue_list,
        batch_size=num_images,
        capacity=10,
        dynamic_pad=True,
        name="batch")

    tf.import_graph_def(nasnet_graph_def, input_map={"input": images})
    graph = tf.get_default_graph()
    conv_feats = graph.get_tensor_by_name("import/final_layer/Relu:0")
    conv_feats = tf.reshape(conv_feats, [num_images, 121, 4032])

    lstm_model = ModelBuilder(max_caption_len, n_lstm_units, num_images, vocab_size)
    captions = lstm_model.build_test_graph(conv_feats)

    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(save_path))
    #for i in range(num_images):
    captions = sess.run(captions)
    idx_to_word = np.load('idx_to_word.npy').tolist()
    for caption in captions:
        words_actual = []
        for idx in caption:
            if (idx == 2):
                break
            word = idx_to_word[idx]
            words_actual.append(word)
        print(words_actual)

train()




