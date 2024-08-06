#!/usr/bin/env python
# -- coding: utf-8 --

import rospy
from octo_ur.msg import Action, Observation

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as Im

class Ur5e:
    def __init__(self):
        self.obs_sub = rospy.Subscriber('/Observation', Observation, self.callback_obs, queue_size=1)
        self.obs = {}

    def callback_obs(self, msg):
        self.obs = {}

        image_pixels = msg.image
        image = []
        for i in image_pixels:
            pixel = [i.x, i.y, i.z]
            image.append(pixel)

        self.obs["image_primary"] = np.array(image).reshape(256,256,3)

        robot_grip = np.ravel(np.array([msg.grip]))
        robot_blocked = np.ravel(np.array([0.0]))
        robot_joint = np.ravel(np.array([msg.joint]))
        robot_pos = np.ravel(np.array([msg.pose.translation.x, msg.pose.translation.y, msg.pose.translation.z]))
        robot_rot = np.ravel(np.array([msg.pose.rotation.x, msg.pose.rotation.y, msg.pose.rotation.z, msg.pose.rotation.w]))

        robot_state = np.concatenate((robot_joint, robot_pos, robot_rot, robot_grip, robot_blocked))

        self.obs["proprio"] = robot_state


def main():
    rospy.init_node("Octo")
    rate = rospy.Rate(10)

    ur5e = Ur5e()

    while not rospy.is_shutdown():
        print(ur5e.obs)
        if "image_primary" in ur5e.obs.keys():
            image = Im.fromarray(ur5e.obs["image_primary"].astype(np.uint8))
            plt.imshow(image)
            plt.pause(1e-9) 
        rate.sleep()

    
if __name__ == "__main__":
    main()     