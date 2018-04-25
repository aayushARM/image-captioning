
from .ModelBuilder import ModelBuilder
from .utils import *
import numpy as np
import time
import json
from io import BytesIO
from tensorflow.python.lib.io import file_io

gcloud_bucket = 'gs://image-captioning-196706-bucket/'
nasnet_path = gcloud_bucket + 'data/frozen_nasnet-large.pb'
save_path = gcloud_bucket + 'saved_checkpoints/'

reader = tf.TFRecordReader()
min_queue_examples = 4600 # ~num of records/shard
num_reader_threads = 3
num_preprocess_threads = 6
max_caption_len = 23
vocab_size = 10204  # 10329
n_lstm_units = 512

#NOTE: In-built python file i/o functions do not work well with GCS Bucket, hence tf.gfile is used wherever needed
with tf.gfile.Open(nasnet_path, mode='rb') as f:
    fileContent = f.read()
    nasnet_graph_def = tf.GraphDef()
    nasnet_graph_def.ParseFromString(fileContent)

def train():
    #Declare necessary constants
    file_names = tf.gfile.Glob(gcloud_bucket + 'MSCOCO_Preprocessed/train*')
    #file_names = glob.glob('/media/aayush/Local Disk/' + 'MSCOCO_Preprocessed/train*')
    batch_size = 32
    initial_learning_rate = 0.0001
    learning_rate=tf.constant(initial_learning_rate)
    total_pairs = 587605 #591435
    num_epochs_per_decay = 0.7 # for exponential decay. Intuitively decided, not fixed.
    clip_gradients = 5.0 # to avoid exploding gradients problem(esp. initially, when loss is very high.
                         # Max value of gradients will be limited to 5.0
    num_batches_per_epoch = int(np.ceil(total_pairs / batch_size))
    num_epochs = 100 # or, till loss saturates.
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

    print('Building input pipeline...\n')
    images, sentences, masks = build_inputs(file_names, reader, min_queue_examples, num_reader_threads,
                                                  num_preprocess_threads, batch_size)

    # Add remaining necessary padding
    caption_len = tf.shape(masks)[1]
    paddings = [[0, 0], [0, 23-caption_len]] # 21 LSTM Steps + 1 start token + 1 end token
    sentences = tf.pad(sentences, paddings, "CONSTANT")
    masks = tf.pad(masks, paddings, "CONSTANT")

    print('Building Nasnet inference graph...\n')
    tf.import_graph_def(nasnet_graph_def, input_map={"input": images})
    graph = tf.get_default_graph()
    conv_feats = graph.get_tensor_by_name("import/final_layer/Relu:0") #Get feats of NASNet's last ReLU layer,
                                                                       #shape: [b_size, 11, 11, 4032]
    conv_feats = tf.reshape(conv_feats, [batch_size, 121, 4032])

    print('Building training LSTM graph...\n')
    lstm_model = ModelBuilder(max_caption_len, n_lstm_units, batch_size, vocab_size)
    batch_loss= lstm_model.build_train_graph(conv_feats, sentences, masks)

    global_step = tf.Variable(140715, trainable=False )#Will be incremeted after every batch is processed

    # exponentially decay learning rate by multiplying it with 0.9 after every decay_steps no. of steps
    def decay_function(learning_rate, global_step):
        return tf.train.exponential_decay(learning_rate, global_step,
                                           decay_steps, 0.90, staircase=True)

    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        adam_optimizer = tf.train.AdamOptimizer(initial_learning_rate)

        #build final op to be run, adam_optimizer.minimize() method can be used as well.
        train_op = tf.contrib.layers.optimize_loss(
            loss=batch_loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=adam_optimizer,
            clip_gradients=clip_gradients,
            learning_rate_decay_fn=decay_function)

    sess=tf.InteractiveSession()
    coord = tf.train.Coordinator()
    #NOTE: Very important op below. Not starting queue runner makes i/p pipeline reader process to go
    #      into a permanent blocked state, with ZERO CPU/GPU usage. Learnt it the hard way ;(
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

#For generating caps for test images
def eval():
    #file_names = tf.gfile.Glob(gcloud_bucket+'val2017/*')
    file_names = tf.gfile.Glob(gcloud_bucket + 'val2014/*')
    num_images = len(file_names)
    batch_size = 166
    num_batches = int(num_images / batch_size)

    #Build a small pipeline to read images in batches from bucket
    filename_queue = tf.train.string_input_producer(file_names, shuffle=False)

    reader = tf.WholeFileReader()
    image_path, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    enqueue_list=[]
    enqueue_list.append([image, image_path])

    images, image_paths = tf.train.batch_join(
        enqueue_list,
        batch_size=batch_size,
        capacity=500,
        dynamic_pad=True,
        name="batch")

    #Similar to training graph:
    tf.import_graph_def(nasnet_graph_def, input_map={"input": images})
    graph = tf.get_default_graph()
    conv_feats = graph.get_tensor_by_name("import/final_layer/Relu:0")
    conv_feats = tf.reshape(conv_feats, [batch_size, 121, 4032])

    lstm_model = ModelBuilder(max_caption_len, n_lstm_units, batch_size, vocab_size)
    captions = lstm_model.build_test_graph(conv_feats)

    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(save_path))
    f = BytesIO(file_io.read_file_to_string(gcloud_bucket + 'data/idx_to_word.npy', binary_mode=True))
    idx_to_word = np.load(f).tolist()

    results = []
    print('Started...')
    for i in range(num_batches):
        int_captions, img_paths = sess.run([captions, image_paths])
        for int_caption, img_path in zip(int_captions, img_paths):
            caption = ''
            cap_len=len(int_caption)
            for j in range(len(int_caption)):
                word = idx_to_word[int_caption[j]]
                if (j+1 != cap_len) and (int_caption[j+1] != 2): #index of end-token = 2
                    caption += word + ' '
                else:
                    caption += word
                    break
            #explicitly add a period if not present
            if caption[-1] != '.':
                caption += ' .'

            #small hack to find image id from path:
            #img_id = img_path[45: 57].decode("utf-8") #decode to avoid binary string

            img_id = img_path[57: 69].decode("utf-8")  # decode to avoid binary string
            img_id = int(img_id.lstrip('0'))

            results.append({'image_id': img_id,
                            'caption': caption})
        print(str((i+1)*batch_size) + '/40670 images processed.')

    print('All images processed.')
    fp = tf.gfile.Open(gcloud_bucket + 'results_val_2014.json', 'wb')
    json.dump(results, fp)

#For Feezing chkpt weights with GraphDef
def export_frozen_model():

    image = tf.placeholder(dtype=tf.uint8, name='processed_image') #First tensor in frozen graph, input.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32, name='final_image')
    image = tf.expand_dims(image, 0)

    tf.import_graph_def(nasnet_graph_def, input_map={"input": image})
    graph = tf.get_default_graph()
    conv_feats = graph.get_tensor_by_name("import/final_layer/Relu:0")
    conv_feats = tf.reshape(conv_feats, [1, 121, 4032]) #b_size = 1 for inference

    lstm_model = ModelBuilder(max_caption_len, n_lstm_units, 1, vocab_size)

    int_caption = lstm_model.build_test_graph(conv_feats)
    int_caption = tf.identity(int_caption[0], 'int_caption') #Final tensor to be taken as output.

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(save_path))

    #Freeze and save to disk
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        ['int_caption'])

    output_graph = "./data/frozen_final_graph.pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

eval()


