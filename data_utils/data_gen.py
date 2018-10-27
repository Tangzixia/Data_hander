#coding=utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import cv2


def _init64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def tfrecordGen(que_dirs):
#     sec_dirs=os.listdir(que_dirs)
#     for index,sec_dir in enumerate(sec_dirs):
#         abs_path=os.path.join(que_dirs,sec_dir)
#         filename="./tf_records/data.tfrecords-{0}".format(index)
#         writer=tf.python_io.TFRecordWriter(filename)
#         for i,file in enumerate(os.listdir(abs_path)):
#             abs_file=os.path.join(abs_path,file)
#             img=cv2.imread(abs_file)
#             res=cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
#             img_raw=res.tobytes()
#             label=i
#             example=tf.train.Example(features=tf.train.Features(feature={
#                 "img_raw":_byte_feature(img_raw),
#                 "label":_init64_feature(label)
#             }))
#             writer.write(example.SerializeToString())
#         writer.close()

def tfrecordGen(que_dirs):
    count=0
    for root,dirs,files in os.walk(que_dirs):
        if len(files)>0:
            file_name="./tf_records/data.tfrecords-{0}".format(count)
            count=count+1
            writer=tf.python_io.TFRecordWriter(file_name)
            for i,file in enumerate(files):
                abs_file=os.path.join(root,file)
                img=cv2.imread(abs_file)
                res=cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
                img_raw=res.tobytes()
                label=i
                example=tf.train.Example(features=tf.train.Features(feature={
                    "img_raw":_byte_feature(img_raw),
                    "label":_init64_feature(label)
                }))
                writer.write(example.SerializeToString())
            writer.close()
if __name__=="__main__":
    que_dirs="../data"
    tfrecordGen(que_dirs)
