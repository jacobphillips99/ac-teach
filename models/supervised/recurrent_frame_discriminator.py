import imp
import tensorflow as tf
import os
import numpy as np

deepknight_root = os.environ.get('DEEPKNIGHT_ROOT')
AbstractModel = imp.load_source("abstractmodel", os.path.join(deepknight_root, "models/model.py")).AbstractModel

class Model(AbstractModel):

    def __init__(self,
                 sess=tf.Session(),
                 trainable=False,
                 CONFIG=None,
                 VS=None):

        self.summary_steps = 0

        INPUT_OBS_SHAPE = [None, 28]
        self.rnn_units = 64
        self.sess = sess

        self.x = tf.placeholder(tf.float32, shape=INPUT_OBS_SHAPE, name='obs') #input frames
        self.y = tf.placeholder(tf.float32, shape=[None,4], name='agent_steering') #agent control
        self.z = tf.placeholder(tf.float32, shape=[None, 1, 1], name='prediction_label') #label of source
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

        self.x_y = tf.concat([self.x, self.y], axis=1)
        # self.obs_features = tf.map_fn(self.obs_encoder, (self.x, self.y), dtype=tf.float32)
        self.obs_features = tf.map_fn(self.obs_encoder, self.x_y, dtype=tf.float32)
        self.rnn_cell = tf.contrib.rnn.LSTMCell(self.rnn_units, forget_bias=1.0) #internal rnn units 64
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.rnn_cell, self.obs_features, dtype=tf.float32)
        self.z_ = tf.map_fn(self.classifier, self.rnn_outputs) #outputs are logits
        self.z_sigmoid = tf.math.sigmoid(self.z_) #prediction sigmoided to [0,1]
        self.z_sigmoid = tf.identity(self.z_sigmoid, name='z_sigmoid')

        if trainable:
            magnitude = 10e4

            # entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.z, logits=self.z_[-2], name="cross_entropy") #using just N-2th frame
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.z, logits=self.z_, name="cross_entropy") #weighting across all frames [:-2]
            w_entropy = entropy * tf.linspace(0.0, 1.0, tf.shape(entropy)[0])
            self.loss = magnitude*tf.reduce_mean(w_entropy)
            optimizer = tf.train.AdamOptimizer(1e-4)
            self.train_step = optimizer.minimize(self.loss)

    def obs_encoder(self, x_y):
        x_y = tf.reshape(x_y, [1,32])
        dense1 = tf.layers.dense(x_y, units=100, activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, units=100, activation=tf.nn.relu)
        flat_with_y = tf.layers.flatten(dense2)
        # flat_with_y = tf.concat([flat, y], axis=1)
        return flat_with_y

    def classifier(self, rnn_output):
        h_fc1 = tf.layers.dense(rnn_output, units=64, activation=tf.nn.relu)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        h_fc2 = tf.layers.dense(h_fc1_drop, units=32, activation=tf.nn.relu)
        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

        z_ = tf.layers.dense(h_fc2_drop, units=1, activation=None)
        z_ = tf.identity(z_, name='logit') #logit
        return z_

    def crop(self, x, roi):
        (i1,j1,i2,j2) = roi
        return x[:, i1:i2, j1:j2, :]

    ''' These are required functions that must be part
        of every model class definition '''

    def init_summaries(self, logdir):
        # create a summary to monitor cost tensor
        discrim_loss = tf.summary.scalar("discrim_loss", self.loss)
        discrim_z_sigmoid = tf.summary.histogram("/label/pred", self.z_sigmoid)   #sigmoid)
        discrim_z_true = tf.summary.histogram("/label/true", self.z)
        # discrim_cropped_image = tf.summary.image("discrim cropped image", self.x_aug, max_outputs=1)

        # merge all summaries into a single op
        self.discrim_merged_summary_op = tf.summary.merge([discrim_loss, discrim_z_sigmoid, discrim_z_true,
                            ], name='Discrim merged summary')

        self.summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
        self.summary_iter = 0

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=200)

    def inference(self, feed):
        sigmoid, logit = self.sess.run([self.z_sigmoid, self.z_], feed_dict=feed)
        return sigmoid, logit

    def bayesian_sample(self, feed, N):
        drop = 0.5
        res = np.zeros(N)
        for i in range(N):
            res[i] = self.inference(dict(feed,**{self.keep_prob: drop} ))[0][0][0]

        mu = np.mean(res)
        var = np.var(res)
        #tau = l**2 * (1 - model.p) / (2 * N * model.weight_decay)

        return res, mu, var

    def train_iter(self, feed):
        (_, z, z_sigmoid,loss) = self.sess.run([self.train_step, self.z, self.z_sigmoid, self.loss], feed_dict=feed)
        report = {'z_sigmoid':z_sigmoid, 'loss':loss, 'z_label': z}
        return report

    def write_summary(self, feed):
        # summary = self.discrim_merged_summary_op.eval(feed_dict=feed, session=self.sess)
        # self.summary_writer.add_summary(summary, self.summary_iter)
        # self.summary_iter += 1
        summary = self.sess.run(self.discrim_merged_summary_op, feed_dict=feed)
        self.summary_writer.add_summary(summary, self.summary_iter)
        self.summary_iter += 1

    def write_checkpoint(self, path, name='model'):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, name )
        filename = self.saver.save(self.sess, checkpoint_path)

    def restore(self, path):
        print('restoring')
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, path)

    def gen_vis_mask(self, im, addToChannel=1):
        import numpy as np
        import cv2
        A = self.sess.run([self.h_conv1,self.h_conv2,self.h_conv3,self.h_conv4,self.h_conv5], feed_dict={self.x:[im]})

        patch = self.sess.run(self.x_aug, feed_dict={self.x:[im]})[0]
        means = [np.mean(patch , 2)]
        # import pdb; pdb.set_trace()
        for i in range(len(A)): #for each feature map
            means.append( np.mean( A[i][0], 2 ) )

        for i in range(len(means)-2, -1, -1):
            smaller = means[i+1]
            scaled_up = cv2.resize(smaller, (means[i].shape[::-1]))
            means[i] = np.multiply(means[i],scaled_up)

        mask = means[0]
        mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))
        mask *= 0.07/mask.mean()
        mask = np.clip(mask, 0,1)

        merged = patch/255.
        for i in range(im.shape[2]):
            merged[:,:,i] += mask * (1 if i==addToChannel else -1)

        return mask, merged

    def create_discrim_feed_inference(self, states, actions):
        #example: states is (34, 74, 192, 3)
        #example: actions is (34, 1)
        feed = {
            self.x: states,
            self.y: actions,
            # self.y: [[action] for action in actions],
            # self.z: np.reshape([0], [len(states), 1, 1]), #0 means from RL model
        }
        return feed


    def create_discrim_feed_training(self, rl_data, im_data, keep_prob=1.0):
        rl_agent_xs, rl_agent_ys, _, _ = rl_data.package_data()
        im_agent_xs, im_agent_ys, _, _ = im_data.package_data()
        rl_agent_zs = rl_data.zs = np.zeros([rl_agent_xs.shape[0], 1, 1])
        im_agent_zs = im_data.zs = np.ones([im_agent_xs.shape[0], 1, 1])

        rl_discrim_feed = {
            self.x: rl_agent_xs,
            self.y: rl_agent_ys,
            self.z: rl_agent_zs,
            self.keep_prob: 1.0,
        }

        im_discrim_feed = {
            self.x: im_agent_xs,
            self.y: im_agent_ys,
            self.z: im_agent_zs,
            self.keep_prob: 1.0,
        }
        return rl_discrim_feed, im_discrim_feed

    def run_discrim_training(self, rl_data, im_data):
        rl_discrim_feed, im_discrim_feed = self.create_discrim_feed_training(rl_data, im_data)
        self.write_summary(rl_discrim_feed)
        rl_discrim_report = self.train_iter(rl_discrim_feed)
        self.write_summary(im_discrim_feed)
        im_discrim_report = self.train_iter(im_discrim_feed)
        return rl_discrim_report, im_discrim_report

    def analyze_discrim_report(self, discrim_report_rl, discrim_report_il):
        rl_last_guess = discrim_report_rl['z_sigmoid'][-1]
        rl_true = discrim_report_rl['z_label'][-1]
        rl_correct = (rl_last_guess < 0.5 and rl_true<0.5) or (rl_last_guess > 0.5 and rl_true>0.5)

        il_last_guess = discrim_report_il['z_sigmoid'][-1]
        il_true = discrim_report_il['z_label'][-1]
        il_correct = (il_last_guess < 0.5 and il_true<0.5) or (il_last_guess > 0.5 and il_true>0.5)

        return rl_correct, il_correct
