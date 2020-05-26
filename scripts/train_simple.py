"""
Running simple policy gradient on an agent in PickAndPlace
"""
import random
import argparse
import gym
import imp
from PIL import Image
from rl_with_teachers.envs import *
from datetime import datetime
import tensorflow as tf
import cv2

root = os.environ.get('DEEPKNIGHT_ROOT')

def setup():
    tf.reset_default_graph()
    env = gym.make("PickPlaceRandomGoal-v0")
    sess = tf.Session()
    rl_model = imp.load_source("rl_model", '/home/jacobphillips/sandbox/ac-teach/models/rl/simple_actor.py').Model(sess=sess, trainable=True)
    #recurrent discrim model
    sess.run(tf.global_variables_initializer())
    return env, sess, rl_model

def create_save_log_dirs(rl_model):
    folder_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    rl_log_dir = "./logs/" + folder_time + "/agent/"
    rl_save_dir = "./save/" + folder_time + "/agent/"
    os.makedirs(rl_log_dir)
    os.makedirs(rl_save_dir)
    #copy for discirm

    rl_model.init_saver()
    rl_model.init_summaries(rl_log_dir)
    #copy for discrim
    return rl_log_dir, rl_save_dir

class ActorDataHolder:
    def __init__(self):
	    self.xs = []
	    self.ys = []
	    self.ys_prev = []
	    self.zs = None

    def append(self, x, y, y_prev):
	    self.xs.append(x)
	    self.ys.append(y)
	    self.ys_prev.append(y_prev)

    def reset(self):
	    self.xs = []
	    self.ys = []
	    self.ys_prev = []
	    self.zs = []

    def package_data(self):
        return np.stack(self.xs), np.array(self.ys), np.array(self.ys_prev)

    def sample(self, num_samples, rs):
        stack_xs, array_ys, array_ys_prev = self.package_data()
        actual_num_samples = min(len(stack_xs), num_samples)
        idx = np.random.choice(stack_xs.shape[0], num_samples)
        discounted_ep_rs_norm = self.discount_and_norm_rewards(rs)
        return stack_xs[idx], array_ys[idx], array_ys_prev[idx], discounted_ep_rs_norm[idx]


def run_expert_rollout(env, goal, num_steps=100, log_path=".", render=False):
    ob = env.reset()
    teacher = OptimalPickPlaceAgent(goal=goal)
    done = False
    step = 0
    ret = 0
    obs = []
    expert_actions = []
    for i in range(num_steps):
    # while not done: #done when 100 steps or completed goal (~40 steps)
        expert_action = teacher(ob)
        ob, r, done, _ = env.step(expert_action)
        obs.append(ob)
        expert_actions.append(expert_action)
        ret += r
        step += 1
        if render:
            env.render()
        # A = env.env.sim.render(500, 500, camera_name='external_camera_0')[::1]
        # save_image(A, path=log_path + "{}.png".format(str(step)))
        if done:
            break
    print("Return {} in {} steps.".format(ret, step))
    return obs, expert_actions


def save_image(arr, path):
    """
    Save a numpy array representing an image to disk.
    :param arr: numpy array corresponding to image
    :param path: string corresponding to the path to save it at
    """
    im = Image.fromarray(arr)
    im.save(path)

if __name__ == "__main__":
    env, sess, rl_model = setup()
    rl_log_dir, rl_save_dir = create_save_log_dirs(rl_model)
    GOAL = env.env.goal #goal location  [1.45, .55, 0.425]

    rl_data = ActorDataHolder()
    rs, states, actions = [], [], []
    best_so_far = 0

    for episode_i in range(100000):
        obs = env.reset()
        done = False
        rl_data.reset()
        rs, states, actions =[], [], []
        step = 0
        EVAL = False
        render = True if episode_i%10==0 else False


        action_prev = [0,0,0,1]
        print("Starting episode " + str(episode_i))

        while not done: #or for i steps in episode?
            rl_feed_inf = rl_model.create_rl_feed_dict_inference(obs, action_prev)
            action = rl_model.inference(rl_feed_inf)

            states.append(obs)
            actions.append(action)
            next_obs, reward, done, _ = env.step(action)
            if render:
                env.render()

            rs.append(reward)
            rl_data.append(obs, action, action_prev)

            step += 1

            if done:

                if sum(rs) > best_so_far:
                    best_so_far = sum(rs)

                rl_feed_train = rl_model.create_rl_feed_dict_train(rl_data, rs, keep_prob=1.0)
                sess.run(rl_model.train_step, rl_feed_train)
                rl_model.write_summary(rl_feed_train)
                rl_model.summary_steps += step


                print("Reward of {} in {} steps".format(sum(rs), step))
                print("Best so far is {}\n".format(best_so_far))

            obs = next_obs
            action_prev = action
