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
    discrim_model = imp.load_source("discrim_model", "/home/jacobphillips/sandbox/ac-teach/models/supervised/recurrent_frame_discriminator.py").Model(sess=sess, trainable=True)

    sess.run(tf.global_variables_initializer())
    return env, sess, rl_model, discrim_model


def create_save_log_dirs(rl_model):
    folder_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    rl_log_dir = "./logs/" + folder_time + "/agent/"
    rl_save_dir = "./save/" + folder_time + "/agent/"
    os.makedirs(rl_log_dir)
    os.makedirs(rl_save_dir)
    rl_model.init_saver()
    rl_model.init_summaries(rl_log_dir)

    discrim_log_dir = "./logs/" + folder_time + "/discrim/"
    discrim_save_dir = "./save/" + folder_time + "/discrim/"
    os.makedirs(discrim_log_dir)
    os.makedirs(discrim_save_dir)
    discrim_model.init_saver()
    discrim_model.init_summaries(discrim_log_dir)
    return rl_log_dir, rl_save_dir, discrim_log_dir, discrim_save_dir, folder_time

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

    def extend(self, xs, ys):
        self.xs.extend(xs)
        self.ys.extend(ys)
        self.ys_prev.append([0,0,0,1])
        self.ys_prev.extend(self.ys[:-1])

    def package_data(self):
        return np.stack(self.xs), np.array(self.ys), np.array(self.ys_prev)

    def sample(self, num_samples, rs):
        stack_xs, array_ys, array_ys_prev = self.package_data()
        actual_num_samples = min(len(stack_xs), num_samples)
        idx = np.random.choice(stack_xs.shape[0], num_samples)
        discounted_ep_rs_norm = self.discount_and_norm_rewards(rs)
        return stack_xs[idx], array_ys[idx], array_ys_prev[idx], discounted_ep_rs_norm[idx]


def run_expert_rollout(env, goal, num_steps=100, log_path=".", render=False):
    ob = env.env.reset(repeat=False)
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
    # print("Return {} in {} steps.".format(ret, step))
    return obs, expert_actions

def run_SLOW_expert_rollout(teacher, env, goal, num_steps=100, log_path=".", render=False):
    env.reset()
    ob = env.env.reset(repeat=False)
    done = False
    step = 0
    ret = 0
    obs = []
    expert_actions = []
    for i in range(num_steps):
    # while not done: #done when 100 steps or completed goal (~40 steps)
        expert_action = teacher(ob)
        expert_action_1, expert_action_2 = split_expert_action(expert_action)
        ob, r, done, _ = env.step(expert_action_1)

        obs.append(ob)
        expert_actions.append(expert_action)
        ret += r
        step += 1

        # env.render()
        # if render:
        #    env.render()
        # A = env.env.sim.render(500, 500, camera_name='external_camera_0')[::1]
        # save_image(A, path=log_path + "{}.png".format(str(step)))
        if done:
            break
        ob, r, done, _ = env.step(expert_action_2)
        obs.append(ob)
        expert_actions.append(expert_action)
        ret += r
        step += 1
    # print("Expert finished in {} steps.".format(step))
    return obs, expert_actions

def split_expert_action(expert_action):
    gripper = [expert_action[3]]
    p = np.random.rand(3)
    q = 1-p
    action_1 = p*expert_action[:3]
    action_2 = q*expert_action[:3]
    return np.concatenate((action_1, gripper)), np.concatenate((action_2,gripper))


def repeat_rollout(rl_model, env, actions):
    env.reset()
    obs = env.env.reset(repeat=True)
    done = False
    images = []
    action_prev = [0,0,0,1]
    # while not done:
        # rl_feed_inf = rl_model.create_rl_feed_dict_inference(obs, action_prev)
        # action = rl_model.inference(rl_feed_inf)
    for action in actions:
        next_obs, reward, done, _ = env.step(action)
        # rend = env.render(mode='rgb_array')
        # env.render()
        rend = env.env.sim.render(500, 500, camera_name='external_camera_0')[::1]
        # import pdb; pdb.set_trace()
        images.append(rend)
    return images




def save_image(arr, path):
    """
    Save a numpy array representing an image to disk.
    :param arr: numpy array corresponding to image
    :param path: string corresponding to the path to save it at
    """
    im = Image.fromarray(arr)
    im.save(path)

if __name__ == "__main__":
    env, sess, rl_model, discrim_model = setup()
    rl_log_dir, rl_save_dir, discrim_log_dir, discrim_save_dir, folder_time = create_save_log_dirs(rl_model)
    GOAL = env.env.goal #goal location  [1.45, .55, 0.425]

    rl_data = ActorDataHolder()
    im_data = ActorDataHolder()
    rs, states, actions = [], [], []
    best_so_far = -9999999990
    teacher = OptimalPickPlaceAgent(goal=GOAL)
    const = 0.9
    print("starting training")

    for episode_i in range(1000000):
        env.reset()
        obs = env.env.reset(repeat=False)
        done = False
        rl_data.reset()
        im_data.reset()
        scores, states, actions, rs = [], [], [], []
        renders = []
        step = 0
        EVAL = False
        render = True if episode_i%100==0 else False

        actor_action_count = 0
	

        action_prev = [0,0,0,1]
        # print("Starting episode " + str(episode_i))

        while not done: #or for i steps in episode?
            if np.random.rand() > const or episode_i%10==0:
                rl_feed_inf = rl_model.create_rl_feed_dict_inference(obs, action_prev)
                action = rl_model.inference(rl_feed_inf)
                actor_action_count += 1
            else:
                action = teacher(obs)

            states.append(obs)
            actions.append(action)
            next_obs, reward, done, _ = env.step(action)
            rs.append(reward)

            # if render:
            #     env.render()

            # rs.append(reward)
            rl_data.append(obs, action, action_prev)

            step += 1

            if done:
                #get evaluation from discrim model
                discrim_feed = discrim_model.create_discrim_feed_inference(states, actions)
                sigmoids, logits = discrim_model.inference(discrim_feed)
                scores.extend([sigmoid[0][0] for sigmoid in sigmoids])
                print("ep " + str(episode_i) + " with return " + str(sum(rs)) + " with " + str(100*float(actor_action_count)/len(rs))[:3] + "% actor agency")

                if sum(rs) > best_so_far and  episode_i%10==0:
                    best_so_far = sum(rs)
                    print("EPISODE {} BEST SO FAR! SAVING CHECKPOINT AND IMAGES".format(episode_i))
                    print("RS WAS BEST ")
                    print(rs)
                    actor_percent = str(100*float(actor_action_count)/len(rs))[:5]
                    print("ACTOR ACTION % IS " + actor_percent)

                    #repeat the last env to get video !!!
                    images = repeat_rollout(rl_model, env, actions)
                    path = "./videos/" + folder_time + "/" + str(episode_i)+'_actor_' + actor_percent + '%'
                    os.makedirs(path)
                    for steps, image in enumerate(images):
                       save_image(image, path=path+ "/{}.png".format(str(steps)))


                    rl_model.write_checkpoint(rl_save_dir, name="model-{}".format(int(sum(rs))))
                    discrim_model.write_checkpoint(discrim_save_dir, name="model-{}".format(episode_i))

                    print("Reward of {} in {} steps".format(sum(rs), step))
                    print("Best so far: {}".format(best_so_far))
                    print("Discrim score of {}".format(scores[-1]))


                if episode_i % 10 == 0:
                    print("\nEVAL EPISODE " + str(episode_i) + "with return " + str(sum(rs))+ "\t with best so far " + str(best_so_far)+ "\n")

                #train RL agent with discrim reward
                rl_feed_train = rl_model.create_rl_feed_dict_train(rl_data, scores, keep_prob=1.0)
                sess.run(rl_model.train_step, rl_feed_train)
                rl_model.write_summary(rl_feed_train)
                rl_model.summary_steps += step
                # print("Reward of {} in {} steps".format(sum(rs), step))
                # print("Best so far: {}".format(best_so_far))
                # print("Discrim score of {}".format(scores[-1]))


                #discrim model training
                expert_obs, expert_actions = run_SLOW_expert_rollout(teacher, env, GOAL, render=render)
                im_data.extend(expert_obs, expert_actions)
                discrim_report_rl, discrim_report_il = discrim_model.run_discrim_training(rl_data, im_data)
                rl_correct, il_correct = discrim_model.analyze_discrim_report(discrim_report_rl, discrim_report_il)
                # print("Discriminator performance on episode: {} \t RL correct: {} \t IL correct: {}\n".format(episode_i, rl_correct, il_correct))

                states, actions = [], []



            obs = next_obs
            action_prev = action
