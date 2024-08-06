#!/usr/bin/env python
# -- coding: utf-8 --

import rospy
from octo_ur.msg import Action, Observation

import gym
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as Im


class Ur5eEnv(gym.Env):
    def __init__(self):
        self.observation_sub = rospy.Subscriber('/Observation', Observation, self.callback_obs, queue_size=1)
        self.observation = {"image_primary": np.zeros((256,256,3)), "proprio": np.zeros((15,))}
        self.isaac_start = False

        self.action_pub = rospy.Publisher('/Action', Action, queue_size=1)

        im_size = 256

        self.observation_space = gym.spaces.Dict(
            {
                **{
                    "image_primary": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((15,)) * -6.28, high=np.ones((15,)) * 6.28, dtype=np.float32
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=np.ones((7,)) * -1/15, high=np.ones((7,)) * 1/15, dtype=np.float32
        )

        self._im_size = im_size
        self._rng = np.random.default_rng(1234)

    def callback_obs(self, msg):
        self.observation = {"image_primary": np.zeros((256,256,3)), "proprio": np.zeros((15,))}

        image_pixels = msg.image
        image = []
        for i in image_pixels:
            pixel = [i.x, i.y, i.z]
            image.append(pixel)

        self.observation["image_primary"] = np.array(image).reshape(256,256,3)

        robot_grip = np.ravel(np.array([msg.grip]))
        robot_blocked = np.ravel(np.array([0.0]))
        robot_joint = np.ravel(np.array([msg.joint]))
        robot_pos = np.ravel(np.array([msg.pose.translation.x, msg.pose.translation.y, msg.pose.translation.z]))
        robot_rot = np.ravel(np.array([msg.pose.rotation.x, msg.pose.rotation.y, msg.pose.rotation.z, msg.pose.rotation.w]))

        robot_state = np.concatenate((robot_joint, robot_pos, robot_rot, robot_grip, robot_blocked))

        self.observation["proprio"] = robot_state

        self.isaac_start = True

    def reset(self):
        # reset_action = Action()
        # reset_action.grip = 10000
        # self.action_pub.publish(reset_action)
        return self.observation, self.observation

    def step(self, action):
        step_action = Action() 

        step_action.grip = action[0]
        step_action.pos.x = action[1]
        step_action.pos.y = action[2]
        step_action.pos.z = action[3]
        step_action.orn.x = action[4]
        step_action.orn.y = action[5]
        step_action.orn.z = action[6]

        self.action_pub.publish(step_action)

        return self.observation, 1, False, False, self.observation

    def get_task(self):
        return {
            "language_instructrion": ["Put the ranch bottle into the pot."]
        }


gym.register(
    "ur5e-sim-isaac-v0",
    entry_point=lambda: Ur5eEnv(),
)
    

# def main():
#     rospy.init_node("Octo")
#     rate = rospy.Rate(10)

#     while not rospy.is_shutdown():
#         print(ur5e.obs)
#         if "image_primary" in ur5e.obs.keys():
#             image = Im.fromarray(ur5e.obs["image_primary"].astype(np.uint8))
#             plt.imshow(image)
#             plt.pause(1e-9) 
#         rate.sleep()

    
# if __name__ == "__main__":
#     main()     