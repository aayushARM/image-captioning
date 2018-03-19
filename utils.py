
import tensorflow as tf

def build_inputs(file_names, reader, min_queue_examples, num_reader_threads, num_preprocess_threads, batch_size):

    filename_queue = tf.train.string_input_producer(
        file_names, shuffle=True, capacity=16, name='shard_read_queue')
    capacity = min_queue_examples + 100 * batch_size

    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name='input_queue')

    enqueue_ops = []
    for _ in range(num_reader_threads):

        _, value = reader.read(filename_queue)
        enqueue_op = values_queue.enqueue([value])
        enqueue_ops.append(enqueue_op)

    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
        values_queue, enqueue_ops))

    context_features = {
        "image/image_id": tf.FixedLenFeature([], dtype=tf.int64),
        "image/data": tf.FixedLenFeature([], dtype=tf.string)
    }

    sequence_features = {
        "image/caption": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    list_for_batching = []
    for thread_id in range(num_preprocess_threads):

        serialized_seq_example = values_queue.dequeue()

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_seq_example,
            context_features=context_features,
            sequence_features=sequence_features
        )

        image_data = context_parsed["image/data"]
        caption = sequence_parsed["image/caption_ids"]
        caption_length = tf.shape(caption)[0]
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        #create mask
        caption_length = tf.expand_dims(caption_length, 0)
        indicator = tf.ones(caption_length, dtype=tf.int32)

        list_for_batching.append([image, caption, indicator])

    queue_capacity = (2 * num_preprocess_threads * batch_size)

    images, sentences, masks = tf.train.batch_join(
        list_for_batching,
        batch_size=batch_size,
        capacity=queue_capacity,
        dynamic_pad=True,
        name="batch")

    return images, sentences, masks





