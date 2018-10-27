#coding=utf-8

import tensorflow as tf
import os

def read_file(tfrecord_file_path):
    # filename_queue=tf.train.string_input_producer(string_tensor=tf.train.match_filenames_once(tfrecord_file_path),
    #                                             num_epochs=None,
    #                                             shuffle=True)
    files=os.listdir(tfrecord_file_path)
    filenames=[]
    for file in files:
        filenames.append(os.path.join(tfrecord_file_path,file))
    filename_queue=tf.train.string_input_producer(filenames)
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)

    features=tf.parse_single_example(serialized_example,features={
        "img_raw":tf.FixedLenFeature([],dtype=tf.string),
        "label":tf.FixedLenFeature([],dtype=tf.int64)
    })
    img=tf.decode_raw(features["img_raw"],out_type=tf.uint8)
    image=tf.reshape(img,(224,224,3))

    label=tf.cast(features["label"],dtype=tf.int64)

    image_batch,label_batch=tf.train.shuffle_batch( [image, label],
      batch_size=5,
      num_threads=3,
      capacity=12,
      min_after_dequeue=3)

    return image_batch,label_batch

def main(_):
    tfrecord_file_path="./tf_records2/"
    image_batch,label_batch=read_file(tfrecord_file_path)
    init_op=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        i=0
        for i in range(10):
            image,label=sess.run([image_batch,label_batch])
            print(image.shape)
            print(i)
        coord.request_stop()
        coord.join(threads=threads)

if __name__=="__main__":
    tf.app.run()