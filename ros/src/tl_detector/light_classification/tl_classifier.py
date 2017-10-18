import rospy
from styx_msgs.msg import TrafficLight
import numpy as np
import os
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from utils import label_map_util
from utils import visualization_utils as vis_util

class TLClassifier(object):
    def __init__(self, simulator=True):
        # rospy.init_node('tl_classifier')

        self.simulator = simulator
        self.light_color = TrafficLight.UNKNOWN
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        model_folder = curr_dir + "/trained_model"

        if simulator:
            path_to_ckpt = model_folder + "/sim_model.pb"
        else:
            path_to_ckpt = model_folder + "/real_resnet_model.pb"
            path_to_chunks = model_folder + "/real_resnet_model_chunks"

            if not os.path.isfile(path_to_ckpt):
                self._join_file_chunks(path_to_chunks, path_to_ckpt)

        # path_to_ckpt = model_folder + "/real_model.pb"
        path_to_label = model_folder + "/light_label.pbtxt"
        num_classes = 4

        #loading label map
        label_map = label_map_util.load_labelmap(path_to_label)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        
        #load frozen Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess =  tf.Session(graph=self.detection_graph) 
        
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        self.light_color = TrafficLight.UNKNOWN

        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                                                          feed_dict={self.image_tensor: image_np_expanded})
        
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        min_thereshold = 0.6
        for i in xrange(0,len(classes)):
            if scores[i] > min_thereshold:
                color = self.category_index[classes[i]]["name"]

                if color == "Red":
                    self.light_color = TrafficLight.RED
                elif color == "Yellow":
                    self.light_color = TrafficLight.YELLOW
                elif color == "Green":
                    self.light_color = TrafficLight.GREEN
                else:
                    self.light_color = TrafficLight.UNKUOWN

                # rospy.loginfo("color: " + color)

        return self.light_color


    def _join_file_chunks(self, chunk_directory, output_file):
        if os.path.exists(chunk_directory):
            if os.path.exists(output_file):
                os.remove(output_file)

            chunks = os.listdir(chunk_directory)
            chunks.sort()

            with open(output_file, 'wb') as fout:
                for f_chunk in chunks:
                    chunk_path = os.path.join(chunk_directory, f_chunk)
                    with open(chunk_path, 'rb') as fin:
                        while True:
                            bytes = fin.read(1024*16)
                            if not bytes:
                                break
                            fout.write(bytes)


