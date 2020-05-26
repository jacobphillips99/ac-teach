"""
Running expert control of agents in PickAndPlace
"""

import random
import argparse
import gym
from PIL import Image
from rl_with_teachers.envs import *
from datetime import datetime

def run_normal_rollout(env, goal):
    ob = env.reset()
    teacher = OptimalPickPlaceAgent(goal=goal)
    folder = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    os.mkdir("./videos/" + folder)
    done = False
    num_steps = 0
    ret = 0
    while not done:
        if teacher is not None:
            ac = teacher(ob)
        else:
            ac = env.action_space.sample()
        ob, r, done, _ = env.step(ac)
        ret += r
        print(r)
        num_steps += 1
        #env.render()
        # A = env.env.sim.render(500, 500, camera_name='external_camera_0')[::1]
        # save_image(A, path="./videos/" + folder + "/{}.png".format("z" + str(num_steps)))
    # print("ob: {}".format(ob))
    print("Return {} in {} steps.".format(ret, num_steps))


def save_image(arr, path):
    """
    Save a numpy array representing an image to disk.
    :param arr: numpy array corresponding to image
    :param path: string corresponding to the path to save it at
    """
    im = Image.fromarray(arr)
    im.save(path)

if __name__ == "__main__":
    env = gym.make("PickPlaceRandomGoal-v0")
    GOAL = env.env.goal
    for i in range(10):
        # run_normal_rollout(env, teacher=teacher)
        run_normal_rollout(env, GOAL)
