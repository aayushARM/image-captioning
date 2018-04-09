
import tensorflow as tf

#Input pipeline implementation.
def build_inputs(file_names, reader, min_queue_examples, num_reader_threads, num_preprocess_threads, batch_size):

    shard_read_queue = tf.train.string_input_producer(
        file_names, capacity=6, shuffle=True) #allow network to see at max any 6 shards at once

    #NOTE: Parallel dequeuing may cause no. of records in values_queue to go below min_queue_examples. Deque ops
    #will be blocked in such case, so allow up to 100 extra batches to be enqueued in values_queue.
    capacity = min_queue_examples + 100 * batch_size

    values_queue = tf.RandomShuffleQueue(
        dtypes=[tf.string],
        min_after_dequeue=min_queue_examples,
        capacity=capacity,
        )

    enqueue_ops = []
    #launch num_reader_threads number of threads for reading from shards:
    for _ in range(num_reader_threads):

        _, value = reader.read(shard_read_queue)
        enqueue_op = values_queue.enqueue([value])
        enqueue_ops.append(enqueue_op)

    #add enqueue_ops list to global queue runner pool
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
        values_queue, enqueue_ops))

    #each SequenceExample protobuf in each TFRecord shard has below 4 fields:
    context_features = {
        "image/image_id": tf.FixedLenFeature([], dtype=tf.int64),
        "image/data": tf.FixedLenFeature([], dtype=tf.string)
    }

    sequence_features = {
        "image/caption": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    list_for_batching = []
    #launch num_preprocess_threads number of threads to dequeue from values_queue, preprocess images,
    #and build inputs for batching:
    for _ in range(num_preprocess_threads):

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
        mask = tf.ones(caption_length, dtype=tf.int32)

        list_for_batching.append([image, caption, mask])

    queue_capacity = (2 * num_preprocess_threads * batch_size) #can be lesser

    images, captions, masks = tf.train.batch_join(
        list_for_batching,
        batch_size=batch_size,
        capacity=queue_capacity,
        dynamic_pad=True, #more padding will still be needed, done later in train()
        name="batch")

    return images, captions, masks





