#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray

import math
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_light_cb)
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self._base_waypoints = None
        self._current_pose = None
        self._last_waypoint = None
        self._max_vel = rospy.get_param("/waypoint_loader/velocity", 10)
        rospy.loginfo("Velocity: {0}".format(self._max_vel))
        self._min_tl_distance = 100
        self._traffic_light_wp = None
        self._distance_to_tl = None
        self.loop()

    def traffic_light_cb(self, msg):
        if self._last_waypoint is not None and msg.data > 0:
            self._traffic_light_wp = msg.data
            dist = self.distance(self._base_waypoints, self._last_waypoint, self._traffic_light_wp)
            self._distance_to_tl = dist
        else:
            self._distance_to_tl = None

    def loop(self):
        rate = rospy.Rate(5) # 5 Hz
        while not rospy.is_shutdown():
            if self._base_waypoints is not None and self._current_pose is not None:
                # publish only after we are initialized
                self.publish_waypoints()
            rate.sleep()

    def publish_waypoints(self):
        # we now need to find the waypoint closest and in-front of this waypoint and add
        # waypoint after that to future waypoints, which in turn is published
        future_waypoints = []
        min_wp = None
        num_base_waypoints = len(self._base_waypoints)
        if self._last_waypoint is None:
            # we are starting. so look through the entire base-waypoints list
            min_wp = self.minDistanceWayPoint(0, num_base_waypoints, self._base_waypoints)
        else:
            # only look through LOOKAHEAD_WPS from the past location
            # assuming that the car won't run through all the way points
            # will need to increase the frequency if it does
            min_wp = self.minDistanceWayPoint(self._last_waypoint, self._last_waypoint + LOOKAHEAD_WPS, self._base_waypoints)

        for i in range(min_wp, min_wp + LOOKAHEAD_WPS):
            wp = self._base_waypoints[i % num_base_waypoints]
            if self._distance_to_tl is not None and self._traffic_light_wp is not None and self._distance_to_tl < self._min_tl_distance:
                # linearly reduce velocity till the traffic light
                # distance and hence velocity would be zero beyond the traffic light
                # dist = self.distance(self._base_waypoints, i%num_base_waypoints, self._traffic_light_wp)
                dist = max(0, self._distance_to_tl / self._min_tl_distance)
                wp.twist.twist.linear.x = dist * self._max_vel
                # rospy.loginfo('{0}: TL: {1}, distance to TL: {2}, velocity = {3}'.format(i, self._traffic_light_wp, dist, wp.twist.twist.linear.x))
            else:
                wp.twist.twist.linear.x = self._max_vel
            future_waypoints.append(wp)
        self._last_waypoint = min_wp
        pubMsg = Lane()
        pubMsg.header.frame_id = '/world'
        pubMsg.header.stamp = rospy.Time(0)
        pubMsg.waypoints = future_waypoints
        self.final_waypoints_pub.publish(pubMsg)
        # rospy.loginfo("Waypoints published.")

    def pose_cb(self, msg):
        self._current_pose = msg

    def minDistanceWayPoint(self, start, end, waypoints, ref_pose = None):
        pose = self._current_pose if ref_pose is None else ref_pose
        min_wp = start
        min_dist = self.distancePose(pose, waypoints[start].pose)
        base_len = len(waypoints)
        for i in range(start + 1, end):
            dist = self.distancePose(pose, waypoints[i % base_len].pose)
            if(dist < min_dist):
                min_dist = dist
                min_wp = i % base_len
        return min_wp if self.is_waypoint_ahead(min_wp, waypoints) else (min_wp + 1)%len(waypoints)

    def is_waypoint_ahead(self, wp, waypoints):
        pose = self._current_pose.pose
        x = pose.position.x
        y = pose.position.y
        _, _, yaw = tf.transformations.euler_from_quaternion([
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # we now translate the way point and car pose into car co-ordinates and check if
        # the way point is infront of us
        wp_pose = waypoints[wp].pose.pose
        wp_x = wp_pose.position.x
        wp_y = wp_pose.position.y

        x_car = (wp_x - x)*math.cos(0 - yaw) - (wp_y - y)*math.sin(0 - yaw)
        # rospy.loginfo('car:{0}, wp: {1}, delta_x: {2}'.format((x, y), (wp_x, wp_y), x_car))
        return (x_car > 0)

    def distancePose(self, pose1, pose2):
        x = pose1.pose.position.x - pose2.pose.position.x
        y = pose1.pose.position.y - pose2.pose.position.y
        z = pose1.pose.position.z - pose2.pose.position.z
        return math.sqrt(x**2 + y**2 + z**2)

    def waypoints_cb(self, waypoints):
        self._base_waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
