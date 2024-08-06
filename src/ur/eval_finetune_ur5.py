#!/usr/bin/env python
# -- coding: utf-8 --

import rospy

import time

from absl import app, flags, logging
import gym
import jax
import numpy as np
import wandb

from uur5e_sim_env import Ur5eEnv

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", "/home/choiyj/octo/octo_ur/", "Path to finetuned Octo checkpoint directory."
)


def main():
    rospy.init_node("octo")

    # setup wandb for logging
    wandb.init(name="eval_ur5e", project="octo")

    # load finetuned model
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(FLAGS.finetuned_path)

    env = gym.make("ur5e-sim-isaac-v0")

    # wrap env to normalize proprio
    env = NormalizeProprio(env, model.dataset_statistics)

    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=50)

    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )

    # create task specification --> use model utility to create task dict with correct entries
    language_instruction = env.get_task()["language_instruction"]
    task = model.create_tasks(texts=language_instruction)

    while not env.isaac_start:
        continue

    obs, info = env.reset()

    # run rollout for 400 steps
    images = [obs["image_primary"][0]]
    episode_return = 0.0

    while not rospy.is_shutdown():
        if len(images) > 400:
            break
            
        # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
        actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
        actions = actions[0]

        # step env -- info contains full "chunk" of observations for logging
        # obs only contains observation for final step of chunk
        obs, reward, done, trunc, info = env.step(actions)
        images.extend([obs["image_primary"]])
        episode_return += reward
        if done or trunc:
            break

    print(f"Episode return: {episode_return}")

    # log rollout video to wandb -- subsample temporally 2x for faster logging
    wandb.log(
        {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
    )


if __name__ == "__main__":
    app.run(main)
