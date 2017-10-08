#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

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

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self._base_waypoints = None
        self._current_pose = None
        self._last_waypoint = None
        rospy.spin()

    def pose_cb(self, msg):
        self._current_pose = msg
        # we now need to find the waypoint closest and in-front of this waypoint and add
        # waypoint after that to future waypoints, which in turn is published
        future_waypoints = []
        min_wp = None
        if self._last_waypoint is None:
            # we are starting. so look through the entire base-waypoints list
            min_wp = self.minDistanceWayPoint(0, len(self._base_waypoints), self._current_pose)
        else:
            # only look through LOOKAHEAD_WPS from the past location
            # assuming that the car won't run through all the way points
            # between 2 successive calls to this function
            min_wp = self.minDistanceWayPoint(self._last_waypoint, self._last_waypoint + LOOKAHEAD_WPS, self._current_pose)
        # rospy.loginfo('Waypoint: {0}'.format(min_wp))
        # rospy.loginfo('Waypoint: {0}'.format(self._base_waypoints[min_wp]))
        # rospy.loginfo('Current position: {0}'.format(msg.pose))
        for i in range(min_wp, min_wp + LOOKAHEAD_WPS):
            future_waypoints.append(self._base_waypoints[i % len(self._base_waypoints)])
        self._last_waypoint = min_wp
        pubMsg = Lane()
        pubMsg.header.frame_id = '/world'
        pubMsg.header.stamp = rospy.Time(0)
        pubMsg.waypoints = future_waypoints
        self.final_waypoints_pub.publish(pubMsg)

    def minDistanceWayPoint(self, start, end, pose):
        min_wp = start
        min_dist = self.distancePose(pose, self._base_waypoints[start].pose)
        base_len = len(self._base_waypoints)
        for i in range(start + 1, end):
            dist = self.distancePose(pose, self._base_waypoints[i % base_len].pose)
            if(dist < min_dist):
                min_dist = dist
                min_wp = i % base_len
        return min_wp

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
