# Import Libraries
import os
import cv2
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from google.protobuf import text_format

# Setup Paths
WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

# Label Map
def construct_label_map():
    labels = [{'name': 'Mask', 'id': 1}, {'name': 'NoMask', 'id': 2}]

    with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

# Convert XML files to CSV
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text+".jpg",
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def convert():
    for directory in ['train', 'test']:
        image_path = f'Tensorflow/workspace/images/{directory}'
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(
            'Tensorflow/workspace/annotations/{}labels.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')

# Create TF record
""" Train : python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/train -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/train.record
Test : python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/test -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/test.record """

# Configuration
def config():
    CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
    config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    # Modify pipeline
    pipeline_config.model.ssd.num_classes = 2
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + \
        '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        ANNOTATION_PATH + '/train.record']
    pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + \
        '/label_map.pbtxt'
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        ANNOTATION_PATH + '/test.record']
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
        f.write(config_text)

# Train Model:
""" $ 'python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=5000'
 """

# Load Model from checkpoints
def load_model():
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-9')).expect_partial()
    return detection_model