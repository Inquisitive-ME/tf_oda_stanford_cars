r"""Convert raw Stanford Cars dataset to TFRecord for object_detection.

Example usage:
    python create_stanford_cars_tf_record.py \
        --data_dir=/data/StanfordCars \
        --output_path=stanford_cars_train.tfrecord \
        --set=train \
        --label_map_path=stanford_cars_label_map.pbtxt

    python create_stanford_cars_tf_record.py \
        --data_dir=/data/StanfordCars \
        --output_path=stanford_cars_test.tfrecord \
        --set=test \
        --label_map_path=stanford_cars_label_map.pbtxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import csv
import PIL.Image

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


car_types_special = ["HHR", "PT Cruiser", "Fiat 500", "Lamborghini", "Ferrari"]

car_types_general = ["Hummer", "Sedan", "Coupe", "Hatchback", "Convertible", "Wagon", "SUV", "Crew Cab",
                         "Cargo Van", "Regular Cab", "Extended Cab", "Van", "Minivan", "Quad Cab", "Club Cab",
                         "SuperCab"]

def write_new_pbtxt(path: str):
    id = 1
    car_types = car_types_general + car_types_special
    with open(path, 'w') as the_file:
        for type in car_types:
            the_file.write('item {\n')
            the_file.write('  id: {}\n'.format(int(id)))
            the_file.write("  name: '" + type + "'\n")
            the_file.write('}\n')
            id += 1


specific_car_type_map = {"Acura TL Type-S 2008": "Sedan",
                "Acura Integra Type R 2001": "Coupe",
                "Buick Regal GS 2012": "Sedan",
                "Chevrolet Corvette ZR1 2012": "Coupe",
                "Chevrolet Corvette Ron Fellows Edition Z06 2007": "Coupe",
                "Chevrolet Cobalt SS 2010": "Coupe",
                "Chevrolet TrailBlazer SS 2009": "SUV",
                "Chrysler 300 SRT-8 2010": "Sedan",
                "Dodge Challenger SRT8 2011": "Coupe",
                "Dodge Charger SRT-8 2009": "Sedan",
                "Jaguar XK XKR 2012": "Coupe"}

flags = tf.app.flags

flags.DEFINE_string('data_dir', '', 'Root directory to Stanford Cars dataset. (car_ims is a subfolder)')
flags.DEFINE_string('output_path', 'stanford_cars.tfrecord', 'Path to output TFRecord.')
flags.DEFINE_string('label_map_path', 'stanford_cars_label_map.pbtxt', 'Path to label map proto.')
flags.DEFINE_string('set', 'merged', 'Convert training set, test set, or merged set.')
flags.DEFINE_string('csv', '', 'Converted CSV labels file')
flags.DEFINE_bool('remap', False, 'Remap labels to more general labels')

FLAGS = flags.FLAGS

SETS = ['train', 'test', 'merged']


def dict_to_tf_example(annotation, dataset_directory, label_map_dict, new_label_map_dict = None):
    im_path = str(annotation['relative_im_path'])
    cls = int(annotation['class'])
    x1 = int(annotation['bbox_x1'])
    y1 = int(annotation['bbox_y1'])
    x2 = int(annotation['bbox_x2'])
    y2 = int(annotation['bbox_y2'])

    # read image
    full_img_path = os.path.join(dataset_directory, im_path)

    # read in the image and make a thumbnail of it
    # max_size = 500, 500
    big_image = PIL.Image.open(full_img_path)
    width, height = big_image.size
    # big_image.thumbnail(max_size, PIL.Image.ANTIALIAS)
    # full_thumbnail_path = os.path.splitext(full_img_path)[0] + '_thumbnail.jpg'
    # big_image.save(full_thumbnail_path)

    with tf.gfile.GFile(full_img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    xmin = []
    xmax = []
    ymin = []
    ymax = []

    # calculate box using original image coordinates
    xmin.append(max(0, x1 / width))
    xmax.append(min(1.0, x2 / width))
    ymin.append(max(0, y1 / height))
    ymax.append(min(1.0, y2 / height))

    # set width and height to thumbnail size for tfrecord ingest
    width, height = image.size

    classes = []
    classes_text = []

    label = ''
    for name, val in label_map_dict.items():
        if val == cls:
            label = name
            break

    if new_label_map_dict:
        mapped = False
        for car_type, mapped_type in specific_car_type_map.items():
            if label == car_type:
                label = mapped_type
                mapped = True
                break
        if not mapped:
            for mapped_type in car_types_special:
                if mapped_type.lower() in label.lower():
                    label = mapped_type
                    mapped = True
                    break
        if not mapped:
          for mapped_type in car_types_general:
            if mapped_type.lower() in label.lower():
              label = mapped_type
              mapped = True
              break

        if not mapped:
          print("Could not map: ", label)
          raise Exception
        classes.append(new_label_map_dict[label])
    else:
        classes.append(label_map_dict[label])

    classes_text.append(label.encode('utf8'))

    image_format = b'jpg'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(full_img_path.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(full_img_path.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.formats(SETS))

    train = FLAGS.set
    data_dir = FLAGS.data_dir
    csv_file = FLAGS.csv

    if FLAGS.remap:
        write_new_pbtxt(os.path.join(os.path.dirname(FLAGS.label_map_path), "new_stanford_cars_label_map.pbtxt"))
        new_label_map_dict = label_map_util.get_label_map_dict(os.path.join(os.path.dirname(FLAGS.label_map_path), "new_stanford_cars_label_map.pbtxt"))

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    with open(csv_file) as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            test = int(row['test'])
            if test:
                testset = 'test'
            else:
                testset = 'train'

            if train == 'merged' or train == testset:
                tf_example = dict_to_tf_example(row, data_dir, label_map_dict, new_label_map_dict)
                writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
