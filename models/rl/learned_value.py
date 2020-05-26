import imp
import tensorflow as tf
import os
import numpy as np
import math
from tensorflow.python.ops import math_ops

deepknight_root = os.environ.get('DEEPKNIGHT_ROOT')
AbstractModel = imp.load_source("abstractmodel", os.path.join(
    deepknight_root, "models/model.py")).AbstractModel


class Model(AbstractModel):

    def __init__(self,
                 sess=tf.Session(),
                 trainable=False,
                 CONFIG=None):

        INPUT_OBS_SHAPE = [None, 28]
        self.sess = sess
        self.CONFIG = CONFIG
        self.summary_steps = 0

        self.tf_xs = tf.placeholder(
            tf.float32, shape=INPUT_OBS_SHAPE, name='obs')
        self.tf_ys = tf.placeholder(
            tf.float32, shape=[None, 4], name='control')
        self.tf_rs = tf.placeholder(
            tf.float32, shape=[None, 1], name='rewards')

        self.tf_ys_prev = tf.placeholder(
            tf.float32, shape=[None, 4], name='control_prev')
        self.keep_prob = tf.placeholder(tf.float32)
        self.tf_value_truths = tf.placeholder(tf.float32, shape=[None, 1])
        self.rewards_ep = tf.placeholder(
            tf.float32, shape=[None, 1], name='rewards_ep')

        # fully connected layers## TODO: concat ys prev??
        # self.input_layer = tf.concat([self.tf_xs, self.tf_ys], axis=1)
        self.fc1 = tf.layers.dense(
            self.tf_xs, units=2048, activation=tf.nn.relu)
        self.fc1_drop = tf.nn.dropout(self.fc1, self.keep_prob)

        self.fc2 = tf.layers.dense(
            self.fc1_drop, units=512, activation=tf.nn.relu)
        self.fc2_drop = tf.nn.dropout(self.fc2, self.keep_prob)

        self.quad_normal_params = tf.layers.dense(
            self.fc2_drop, units=8, activation=None)
        self.mus, self.log_sigmas = tf.split(
            self.quad_normal_params, [4, 4], axis=1)
        self.mus = tf.tanh(self.mus)
        self.sigmas = 0.05 * tf.sigmoid(self.log_sigmas) + .001

        self.flat_value_learning = tf.layers.dense(
	    self.fc2_drop, units=2, activation=None)
        self.value_mu, self.value_log_sigma = tf.split(
	    self.flat_value_learning, [1, 1], axis=1)
        self.value_sigma = 0.05 * tf.sigmoid(self.value_log_sigma) + 0.01

        self.dists = tf.distributions.Normal(loc=self.mus, scale=self.sigmas)
        self.joints_sample = self.dists.sample([1])
        self.joints_ = tf.identity(self.mus, name='prediction')

        self.value_dist = tf.distributions.Normal(
            loc=self.value_mu, scale=self.value_sigma)
        self.value_sample = self.value_dist.sample([1])
        self.value_ = tf.identity(self.value_mu, name='value_prediction')

        if trainable:
            self.neg_log_prob = -1 * self.dists.log_prob(self.tf_ys)
            self.product = self.neg_log_prob * self.tf_rs
            self.value_error = tf.reduce_mean(
                tf.square(tf.subtract(self.value_sample, self.tf_value_truths)))
            self.loss = 1.0 * \
                tf.reduce_mean(self.product) + 1.0*self.value_error
            optimizer = tf.train.AdamOptimizer(1e-4)
            self.gradients, variables = zip(
                *optimizer.compute_gradients(self.loss))
            self.gradients, _ = tf.clip_by_global_norm(self.gradients, 5.0)

            self.train_step = optimizer.apply_gradients(
                zip(self.gradients, variables))

    ''' These are required functions that must be part
        of every model class definition '''

    def init_summaries(self, logdir):
        # create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.loss)
        self.gradients, variables = zip(
            *optimizer.compute_gradients(self.loss))
        self.gradients, _ = tf.clip_by_global_norm(self.gradients, 5.0)

        # self.capped_gvs = [(grad, var) for grad, var in self.gvs]
        self.train_step = optimizer.apply_gradients(
            zip(self.gradients, variables))
        # self.train_step = optimizer.minimize(self.loss)

    ''' These are required functions that must be part
        of every model class definition '''

    def init_summaries(self, logdir):
        # create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("neglogprob", tf.reduce_mean(self.neg_log_prob))
        tf.summary.scalar("product", tf.reduce_mean(self.product))
        tf.summary.scalar("rewards", tf.reduce_sum(self.tf_rs))
        self.rewards_ep_summary = tf.summary.scalar(
            "rewards_ep", tf.reduce_mean(self.rewards_ep))
        # tf.summary.scalar("qloss", self.q_loss)
        tf.summary.scalar("ys", tf.reduce_mean(self.tf_ys))
        tf.summary.scalar("value_error", self.value_error)
        # tf.summary.scalar('angle_regularizer', tf.reduce_mean(self.angle_regularizer))
        # tf.summary.histogram('delta', self.delta)
        tf.summary.histogram("global_norm", tf.global_norm(
            [tf.reshape(g, (-1,)) for g in self.gradients]))
        tf.summary.histogram("mu", self.mus)
        tf.summary.histogram("sigma", self.sigmas)
        # tf.summary.histogram("y hat", self.y_sample)
        tf.summary.histogram("y", self.joints_sample)
        # tf.summary.histogram("gradients", tf.stack([tf.reshape(g,(-1,)) for g in self.gradients]))
        # tf.summary.image("obs", self.x_image, max_outputs=1)
        # tf.summary.histogram("last layer", tf.get_default_graph().get_tensor_by_name(os.path.split(self.h_fc2.name)[0] + '/kernel:0'))
        # merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

        self.summary_writer = tf.summary.FileWriter(
            logdir, graph=tf.get_default_graph())
        self.summary_iter = 0

    def init_saver(self):
        self.saver = tf.train.Saver()

    def inference(self, feed):
        joints_sample, value_sample = self.sess.run(
            [self.joints_sample, self.value_sample], feed_dict=feed)
        return joints_sample[0][0], value_sample[0][0]

    def train_iter(self, feed):
        (_, joints, joints_sample, mus, sigmas, loss, product, neg_log_prob, value_sample, value_) = self.sess.run(
            [self.train_step, self.joints_, self.joints_sample, self.mus, self.sigmas, self.loss, self.product, self.neg_log_prob, self.value_sample, self.value_], feed_dict=feed)
        report = {'joints': joints, 'joints_sample': joints_sample, 'mus': mus, 'sigmas': sigmas, 'loss': loss,
            'product': product, 'neg_log_prob': neg_log_prob, 'value_': value_, 'value_sample': value_sample}
        return report

    def write_summary(self, feed):
        summary = self.sess.run(self.merged_summary_op, feed_dict=feed)
        self.summary_writer.add_summary(summary, self.summary_iter)
        self.summary_iter += 1

    def write_checkpoint(self, path, name='model'):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, name)
        filename = self.saver.save(self.sess, checkpoint_path)

    def restore(self, path):
        print('restoring')
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, path)

    def create_rl_feed_dict_inference(self, obs, action_prev, keep_prob=1.0):
        return {
            self.tf_xs: [obs],
            self.tf_ys_prev: [action_prev],
            self.keep_prob: 1.0,
        }

    def create_rl_feed_dict_train(self, rl_data, rs, keep_prob=1.0):
        np_rs = np.array(rs)
        ep_rs_sum = np.sum(np_rs)
        discounted_ep_rs = np.zeros_like(rs)
        running_add = 0
        gamma = 0.95
        for t in reversed(range(0, len(rs))):
            running_add = running_add * gamma + rs[t]
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        discounted_rewards = discounted_ep_rs
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        discounted_ep_rs_norm = discounted_ep_rs.reshape((-1,1))

        return {
            self.tf_xs: rl_data.xs,  # shape=[None, 28]
            self.tf_ys: rl_data.ys,  # shape=[None, 4]
            self.tf_ys_prev: rl_data.ys_prev,
            self.tf_rs: discounted_ep_rs_norm,  # shape=[None, 1]
            self.rewards_ep: [[ep_rs_sum]],
            self.keep_prob: keep_prob,
            self.tf_value_truths: discounted_rewards.reshape((-1,1))
        }

	
