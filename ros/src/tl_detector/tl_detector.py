#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np
from timeit import default_timer as timer

STATE_COUNT_THRESHOLD = 3
DEBUG_MODE = False


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.light_str = {0: "RED", 1: "YELLOW", 2: "GREEN", 4: "UNKNOWN"}

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.current_light_index = None
        self.last_car_wp = None
        self.lights = []
        self.stop_line_waypoints = []
        self.detection_time = 0
        self.start_detection = None
        self.end_detection = None

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        # empty list because of new waypoints, for the following pre-computation
        self.stop_line_waypoints = []
        # compute in advance positions of the stopline waypoints
        for light_posn in self.config['stop_line_positions']:
            pose = Pose()
            pose.position.x = light_posn[0]
            pose.position.y = light_posn[1]
            self.stop_line_waypoints.append(self.get_closest_waypoint(pose))

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        # skip detection to avoid delay due to queueing on the image publisher
        if self.detection_time > 0:
            self.detection_time -= 0.1  # equivalent to 10hz (publication frequency of image_color)
            rospy.logdebug("skip tl detection")
            return

        self.has_image = True
        self.camera_image = msg
        self.start_detection = timer()
        tr_light_wp, tr_light_state = self.process_traffic_lights()
        self.end_detection = timer()
        self.detection_time = self.end_detection - self.start_detection
        rospy.logdebug("image_cb time delay: %f", self.detection_time)

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != tr_light_state:
            self.state_count = 0
            self.state = tr_light_state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            tr_light_wp = tr_light_wp if tr_light_state == TrafficLight.RED or tr_light_state == TrafficLight.YELLOW else -1
            self.last_wp = tr_light_wp
            self.upcoming_red_light_pub.publish(Int32(tr_light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose, start_from=0):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        if self.waypoints is None:
                return None

        nearest_idx = None
        d_min = float("inf")
        for i in range(0, len(self.waypoints.waypoints)):
            if start_from > 0:
                i = (i + start_from) % len(self.waypoints.waypoints)
            ith_waypoint = self.waypoints.waypoints[i]
            d = TLDetector.distance(pose.position, ith_waypoint.pose.pose.position)
            if d < d_min:
                d_min = d
                nearest_idx = i
            if start_from != 0 and d > d_min:
                break
        return nearest_idx

    @staticmethod
    def distance(a, b):
        """Computes distance between two positions a and b
        Args:
            a (position): first position
            b (position): second position

        Returns:
            value: distance value between a and b
        """
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    def get_light_state(self):
        """Predicts the id of the traffic light as captured by camera

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            return False

        cv2_img = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        tl_id_prediction = self.light_classifier.get_classification(cv2_img)

        return tl_id_prediction

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # visibility range
        tl_vis_range = 256

        if self.pose and self.stop_line_waypoints:
            car_posn = self.get_closest_waypoint(self.pose.pose, self.last_car_wp or 0)
            self.last_car_wp = car_posn
            nearest_light_wp = None
            nearest_light_idx = None
            # minimal distance
            d_min = float("inf")
            # find traffic light with minimal distance from the current car position
            for i, i_light in enumerate(self.lights):
                i_light_posn = i_light.pose.pose.position
                posn = self.waypoints.waypoints[car_posn].pose.pose.position
                # calc distance b/w current car position and i-th traffic light
                d = TLDetector.distance(posn, i_light_posn)
                i_light_wp = self.stop_line_waypoints[i]
                if d < d_min and ((0 <= i_light_wp - car_posn < tl_vis_range) or (
                        0 <= i_light_wp + len(self.waypoints.waypoints) - car_posn < tl_vis_range)):
                    d_min = d
                    nearest_light_idx = i
                    nearest_light_wp = i_light_wp

            if nearest_light_wp is not None:
                # store closest light position
                self.current_light_index = nearest_light_idx

                traffic_light_detection_was_implemented = False

                if traffic_light_detection_was_implemented:
                    # when get_light_state() will be properly implemented:
                    return nearest_light_wp, self.get_light_state()
                else:
                    # use dummy/substitute, peeking true light state from the topic: /vehicle/traffic_lights
                    rospy.loginfo("TL: closest_light_wp=%d,  self.current_light_index=%d,  state=%s,  d_min=%d",
                                  nearest_light_wp, self.current_light_index,
                                  self.light_str[self.lights[self.current_light_index].state],
                                  d_min)
                    return nearest_light_wp, self.lights[self.current_light_index].state

        rospy.loginfo("Dummy TL: closest_light_wp=%d,  self.current_light_index=%d,  state=%s", -1, -1,
                      self.light_str[TrafficLight.UNKNOWN])
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
