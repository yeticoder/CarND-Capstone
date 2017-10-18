#!/bin/python

import sys
import os

tf_paths = ['~/tensorflow/tf_models/research', '~/tensorflow/tf_models/research/slim']
for p in tf_paths:
    ap = os.path.expanduser(p)
    if ap not in sys.path:
        sys.path.append(ap)


from PIL import Image
import tensorflow as tf
import yaml

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_file', '', 'Path to output TFRecord')
flags.DEFINE_string('annotations_file', '', 'File annotations')
flags.DEFINE_string('annotations_file2', '', 'File annotations')
# flags.DEFINE_boolean('bosch', False, "Bosch dataset")

FLAGS = flags.FLAGS

LABELS = {
    "Green" : 1,
    "Red" : 2,
    "Yellow" : 3,
    "off" : 4
}

SKIP_THRESHOLD = 10.0
# IMG_SIZE = [800, 600]
# IMG_SIZE = [1368, 1096]
# IMG_SIZE = [1280, 720]

KEYWORDS = ['Left', 'Right', 'Straight']


def normalize_label(label):
    for k in KEYWORDS:
        label = label.replace(k, '')
    return label


def get_tf_example(annotation, base_dir):
    bosch_ds = False
    if "path" in annotation:
        bosch_ds = True

    filename = os.path.join(base_dir, annotation['path'] if bosch_ds else annotation['filename'])

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image = fid.read()

    img = Image.open(filename)
    width, height = img.size

    image_format = os.path.splitext(filename)[1][1:]

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []

    boxes = annotation['boxes'] if bosch_ds else annotation['annotations']

    skip = True

    for a in boxes:
        x_min = a['x_min'] if bosch_ds else a['xmin']
        xmins.append(float(x_min)/width)
        x_max = a['x_max'] if bosch_ds else a['xmin'] + a['x_width']
        xmaxs.append(float(x_max)/width)
        y_min = a['y_min'] if bosch_ds else a['ymin']
        ymins.append(float(y_min)/height)
        y_max = a['y_max'] if bosch_ds else a['ymin'] + a['y_height']
        ymaxs.append(float(y_max)/height)

        if x_max - x_min > SKIP_THRESHOLD:
            skip = False
        # print("xwidth %s ywidth %s %s" % (x_max-x_min, y_max-y_min, annotation['path']))

        class_v = normalize_label(a['label']) if bosch_ds else a['class']
        classes_text.append(class_v.encode('utf8'))
        classes.append(LABELS[class_v])


    # skip very small and empty bosch traffic lights
    if len(boxes) == 0 and (not bosch_ds):
        skip = False

    if skip:
        return None


    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))




def main(_):

    files = []
    if not os.path.isfile(FLAGS.annotations_file):
        raise IOError("Annotations file not found")
    files.append(FLAGS.annotations_file)

    if os.path.isfile(FLAGS.annotations_file2):
        files.append(FLAGS.annotations_file2)

    writer = tf.python_io.TFRecordWriter(FLAGS.output_file)

    for an_file in files:
        with open(an_file, 'r') as f:
            annotations = yaml.load(f)

        print("%s annotations loaded" % len(annotations))

        base_dir = os.path.dirname(an_file)
        c = 0
        for a in annotations:
            tf_example = get_tf_example(a, base_dir)
            if not tf_example:
                continue

            c += 1
            writer.write(tf_example.SerializeToString())

        print("%s images written" % c)

    writer.close()


if __name__ == '__main__':
    tf.app.run()



