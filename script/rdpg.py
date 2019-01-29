import tensorflow as tf
import numpy as np
import os
import copy
import time
import model_utils as model_utils
import matplotlib.pyplot as plt
from tensorflow.python.ops.rnn_cell import LSTMStateTuple

class Actor(object):
    def __init__(self,
                 sess,
                 depth_size,
                 n_hidden,
                 max_steps,
                 learning_rate,
                 batch_size,
                 action_range,
                 tau,
                 n_layers
        ):

        self.sess = sess
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.action_range = action_range
        self.a_dim = 2
        self.tau = tau
        self.n_layers = n_layers
        self.max_steps = max_steps
        self.batch_size = batch_size

        with tf.variable_scope('actor'):
            self.depth_input = tf.placeholder(tf.float32, [None, 
                                                            depth_size[0], 
                                                            depth_size[1], 
                                                            depth_size[2]],
                                                            name='depth_input') # b*l, h, w, c

            self.lengths = tf.placeholder(tf.int32, [self.batch_size], name='lengths') # b
     
            with tf.variable_scope('online'):
                self.a_online, self.a_test_online, self.rnn_state_online, self.prev_rnn_state_online = self.Model()
            self.network_params = tf.trainable_variables()

            with tf.variable_scope('target'):
                self.a_target, self.a_test_target, _, _ = self.Model()
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))] 

        # This gradient will be provided by the critic network
        self.a_gradient = tf.placeholder(tf.float32, [None, self.a_dim]) # b*l, 2

        # Combine the gradients here
        self.gradients = tf.gradients(self.a_online, self.network_params, -self.a_gradient)

        # Optimization Op by applying gradient, variable pairs
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def Model(self):      
        conv1 = model_utils.Conv2D(self.depth_input, 4, (5, 5), (4, 4), scope='conv1') # b*l, h, w, c
        conv2 = model_utils.Conv2D(conv1, 16, (5, 5), (4, 4), scope='conv2') # b*l, h, w, c
        conv3 = model_utils.Conv2D(conv2, 32, (3, 3), (2, 2), scope='conv3') # b*l, h, w, c
        shape = conv3.get_shape().as_list()

        rnn_cell = model_utils._lstm_cell(self.n_hidden, self.n_layers)

        w_linear_a = tf.get_variable('w_linear', [self.n_hidden, 1], initializer=tf.initializers.random_uniform(-0.003, 0.003))
        w_angular_a = tf.get_variable('w_angular', [self.n_hidden, 1], initializer=tf.initializers.random_uniform(-0.003, 0.003))
        b_linear_a = tf.get_variable('b_linear_a', [1], initializer=tf.initializers.random_uniform(-0.003, 0.003))
        b_angular_a = tf.get_variable('b_angular_a', [1], initializer=tf.initializers.random_uniform(-0.003, 0.003))

        # training
        depth_vectors = tf.reshape(conv3, (self.batch_size, self.max_steps, shape[1]*shape[2]*shape[3])) # b, l, h

        rnn_outputs, _ = tf.nn.dynamic_rnn(rnn_cell, 
                                            depth_vectors, 
                                            sequence_length=self.lengths,
                                            dtype=tf.float32) # b, l, h

        rnn_outputs_reshape = tf.reshape(rnn_outputs, [-1, self.n_hidden]) # b*l, h

        a_linear = tf.nn.sigmoid(tf.matmul(rnn_outputs_reshape, w_linear_a) + b_linear_a) * self.action_range[0] # b*l, 1
        a_angular = tf.nn.tanh(tf.matmul(rnn_outputs_reshape, w_angular_a) + b_angular_a) * self.action_range[1] # b*l, 1
        a = tf.concat([a_linear, a_angular], axis=1)# b*l, 2

        # testing
        prev_rnn_state = []
        for l in xrange(self.n_layers):
            prev_rnn_state.append(
                LSTMStateTuple(tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='initial_state1{0}.c'.format(l)),
                               tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='initial_state1{0}.h'.format(l))))
        if self.n_layers == 1:
            prev_rnn_state = prev_rnn_state[0]

        depth_vectors_test = tf.reshape(conv3, (1, 1, shape[1]*shape[2]*shape[3])) # b, l, h

        rnn_outputs_test, rnn_state = rnn_cell(tf.reshape(depth_vectors_test, [-1, shape[1]*shape[2]*shape[3]]), prev_rnn_state)

        a_linear_test = tf.nn.sigmoid(tf.matmul(rnn_outputs_test, w_linear_a) + b_linear_a) * self.action_range[0] # b*l, 1
        a_angular_test = tf.nn.tanh(tf.matmul(rnn_outputs_test, w_angular_a) + b_angular_a) * self.action_range[1] # b*l, 1
        a_test = tf.concat([a_linear_test, a_angular_test], axis=1) # b*l, 2

        return a, a_test, rnn_state, prev_rnn_state

    def Train(self, depth_input, lengths, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.depth_input: depth_input,
            self.lengths: lengths,
            self.a_gradient: a_gradient
            })

    def PredictSeqTarget(self, depth_input, lengths):
        return self.sess.run(self.a_target, feed_dict={
            self.depth_input: depth_input,
            self.lengths: lengths
            })

    def PredictSeqOnline(self, depth_input, lengths):
        return self.sess.run(self.a_online, feed_dict={
            self.depth_input: depth_input,
            self.lengths: lengths
            })

    def Predict(self, depth_input, prev_rnn_state_online):
        return self.sess.run([self.a_test_online, self.rnn_state_online], feed_dict={
            self.depth_input: depth_input,
            self.prev_rnn_state_online: prev_rnn_state_online
            })

    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)

    def TrainableVarNum(self):
        return self.num_trainable_vars



class Critic(object):
    def __init__(self,
                 sess,
                 depth_size,
                 n_hidden,
                 max_steps,
                 learning_rate,
                 batch_size,
                 num_actor_vars,
                 tau,
                 n_layers
        ):

        self.sess = sess
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.tau = tau
        self.n_layers = n_layers
        self.max_steps = max_steps
        self.batch_size = batch_size

        with tf.variable_scope('critic'):

            self.depth_input = tf.placeholder(tf.float32, [None, 
                                                            depth_size[0], 
                                                            depth_size[1], 
                                                            depth_size[2]],
                                                            name='depth_input') # b*l, h, w, c
            self.action_input = tf.placeholder(tf.float32, [None, 2],
                                                            name='action_input') # b*l, 2

            self.lengths = tf.placeholder(tf.int32, [self.batch_size], name='lengths') # b

            with tf.variable_scope('online'):
                self.q_online, self.q_test_online, self.rnn_state_online, self.prev_rnn_state_online = self.Model()
            self.network_params = tf.trainable_variables()[num_actor_vars:]

            with tf.variable_scope('target'):
                self.q_target, self.q_test_target, _, _, = self.Model()
            self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.predicted_q = tf.placeholder(tf.float32, [self.batch_size, self.max_steps, 1], name='predicted_q')
        self.mask = tf.expand_dims(tf.sequence_mask(self.lengths, maxlen=self.max_steps, dtype=tf.float32), axis=2) # b, l, 1
        self.square_diff = tf.pow((self.predicted_q - tf.reshape(self.q_online, (self.batch_size, self.max_steps, 1)))*self.mask, 2) # b, l, 1

        self.loss_t = tf.reduce_sum(self.square_diff, reduction_indices=1) / tf.cast(self.lengths, tf.float32)# b, 1
        self.loss_n = tf.reduce_sum(self.loss_t, reduction_indices=0) / self.batch_size # 1

        self.gradient = tf.gradients(self.loss_n, self.network_params)
        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.opt.apply_gradients(zip(self.gradient, self.network_params))

        mask_reshape = tf.reshape(self.mask, (self.batch_size*self.max_steps, 1)) # b*l, 1
        self.a_gradient_mask = tf.tile(mask_reshape, [1, 2]) # b*l, 2
        self.action_grads = tf.gradients(self.q_online, self.action_input) * self.a_gradient_mask

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))] 


    def Model(self):
        conv1 = model_utils.Conv2D(self.depth_input, 4, (5, 5), (4, 4), scope='conv1') # b*l, h, w, c
        conv2 = model_utils.Conv2D(conv1, 16, (5, 5), (4, 4), scope='conv2') # b*l, h, w, c
        conv3 = model_utils.Conv2D(conv2, 32, (3, 3), (2, 2), scope='conv3') # b*l, h, w, c
        shape = conv3.get_shape().as_list()

        rnn_cell = model_utils._lstm_cell(self.n_hidden, self.n_layers)

        w_q = tf.get_variable('w_q', [self.n_hidden, 1], initializer=tf.initializers.random_uniform(-0.003, 0.003))
        b_q = tf.get_variable('b_q', [1], initializer=tf.initializers.random_uniform(-0.003, 0.003))

        # training
        depth_vectors = tf.reshape(conv3, (self.batch_size, self.max_steps, shape[1]*shape[2]*shape[3]), name='train_d_reshape') # b, l, h*w*c
        action_input_reshape = tf.reshape(self.action_input, (self.batch_size, self.max_steps, 2), name='train_a_reshape') # b, l, 2
        inputs = tf.concat([depth_vectors, action_input_reshape], axis=2) # b, l, h*w*c+2

        rnn_outputs, _ = tf.nn.dynamic_rnn(rnn_cell, 
                                            inputs, 
                                            sequence_length=self.lengths,
                                            dtype=tf.float32) # b, l, h

        rnn_outputs_reshape = tf.reshape(rnn_outputs, [-1, self.n_hidden]) # b*l, h

        q = tf.matmul(rnn_outputs_reshape, w_q) + b_q # b*l, 1
        # q = tf.reshape(q, (self.batch_size, self.max_steps, 1))

        # testing
        depth_vectors_test = tf.reshape(conv3, (1, 1, shape[1]*shape[2]*shape[3]), name='test_d_reshape') # b, l, h*w*c
        action_input_reshape_test = tf.reshape(self.action_input, (1, 1, 2), name='test_a_reshape') # b, l, 2
        inputs_test = tf.concat([depth_vectors_test, action_input_reshape_test], axis=2) # b, l, h*w*c+2

        prev_rnn_state = []
        for l in xrange(self.n_layers):
            prev_rnn_state.append(
                LSTMStateTuple(tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='initial_state1{0}.c'.format(l)),
                               tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='initial_state1{0}.h'.format(l))))
        if self.n_layers == 1:
            prev_rnn_state = prev_rnn_state[0]

        rnn_outputs_test, rnn_state = rnn_cell(tf.reshape(inputs_test, (-1, shape[1]*shape[2]*shape[3]+2)), prev_rnn_state)

        q_test = tf.matmul(rnn_outputs_test, w_q) + b_q # b*l, 1

        return q, q_test, rnn_state, prev_rnn_state

    def Train(self, depth_input, action_input, predicted_q, lengths):
        return self.sess.run([self.q_online, self.optimize], feed_dict={
            self.depth_input: depth_input,
            self.action_input: action_input,
            self.predicted_q: predicted_q,
            self.lengths: lengths
            })

    def PredictSeqOnline(self, depth_input, action_input, lengths):
        return self.sess.run(self.q_online, feed_dict={
            self.depth_input: depth_input,
            self.action_input: action_input,
            self.lengths: lengths
            })

    def PredictSeqTarget(self, depth_input, action_input, lengths):
        return self.sess.run(self.q_target, feed_dict={
            self.depth_input: depth_input,
            self.action_input: action_input,
            self.lengths: lengths
            })

    def Predict(self, depth_input, action_input, prev_rnn_state_online):
        return self.sess.run([self.q_test_online, self.rnn_state_online], feed_dict={
            self.depth_input: depth_input,
            self.action_input: action_input,
            self.prev_rnn_state_online: prev_rnn_state_online,
            })

    def ActionGradients(self, depth_input, action_input, lengths):
        return self.sess.run(self.action_grads, feed_dict={
            self.depth_input: depth_input,
            self.action_input: action_input,
            self.lengths: lengths
            })

    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)



class RDPG(object):
    """docstring for RDPG"""
    def __init__(self, flags, sess):
        self.depth_size = [flags.depth_h, flags.depth_w, flags.depth_c]
        self.n_hidden = flags.n_hidden
        self.a_learning_rate = flags.a_learning_rate
        self.c_learning_rate = flags.c_learning_rate
        self.batch_size = flags.batch_size
        self.max_steps = flags.max_steps
        self.tau = flags.tau
        self.n_layers = flags.n_layers
        self.action_range = [flags.a_linear_range, flags.a_angular_range]
        self.buffer_size = flags.buffer_size
        self.a_dim = flags.a_dim
        self.gamma = flags.gamma

        self.actor = Actor(sess=sess,
                           depth_size=self.depth_size,
                           n_hidden=self.n_hidden,
                           max_steps=self.max_steps,
                           learning_rate=self.a_learning_rate,
                           batch_size=self.batch_size,
                           action_range=self.action_range,
                           tau=self.tau,
                           n_layers=self.n_layers)

        self.critic = Critic(sess=sess,
                             depth_size=self.depth_size,
                             n_hidden=self.n_hidden,
                             max_steps=self.max_steps,
                             learning_rate=self.c_learning_rate,
                             batch_size=self.batch_size,
                             num_actor_vars=len(self.actor.network_params)+len(self.actor.target_network_params),
                             tau=self.tau,
                             n_layers=self.n_layers)
        self.memory = []

    def ActorPredict(self, depth_input, t):
        if t == 0:
            prev_rnn_state_online=(np.zeros([1, self.n_hidden]), np.zeros([1, self.n_hidden]))
        else:
            prev_rnn_state_online = copy.deepcopy(self.rnn_state)
        a, self.rnn_state = self.actor.Predict(depth_input, prev_rnn_state_online)
        return a

    def Add2Mem(self, seq):
        self.memory.append(seq) #seq: ((d_0, a_0, r_0), (d_1, a_1, r_1), ..., (d_T, a_T, r_T))
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def SampleBatch(self):
        if len(self.memory) >= self.batch_size:
            indices = np.random.randint(0, len(self.memory), size=(self.batch_size))
            depth_t_batch = []
            action_batch = []
            reward_batch = []
            lengths_batch = []
            for idx in indices:
                sampled_seq = copy.deepcopy(self.memory[idx])
                seq_len = len(sampled_seq)

                full_depth_t_seq = np.zeros([self.max_steps, self.depth_size[0], self.depth_size[1], self.depth_size[2]])
                full_action_seq = np.zeros([self.max_steps, self.a_dim])
                full_reward_seq = np.zeros([self.max_steps])
                for t in xrange(seq_len):
                    full_depth_t_seq[t, :, :, 0] = sampled_seq[t][0]
                    if t == 0:
                        full_depth_t_seq[t, :, :, 1] = sampled_seq[t][0]  
                        full_depth_t_seq[t, :, :, 2] = sampled_seq[t][0]
                    elif t == 1:
                        full_depth_t_seq[t, :, :, 1] = sampled_seq[t-1][0]
                        full_depth_t_seq[t, :, :, 2] = sampled_seq[t-1][0]
                    else:
                        full_depth_t_seq[t, :, :, 1] = sampled_seq[t-1][0]
                        full_depth_t_seq[t, :, :, 2] = sampled_seq[t-2][0]                   

                    full_action_seq[t, :] = sampled_seq[t][1]
                    full_reward_seq[t] = sampled_seq[t][2]



                depth_t_batch.append(full_depth_t_seq)
                action_batch.append(full_action_seq)
                reward_batch.append(full_reward_seq)
                lengths_batch.append(seq_len)

            shape = self.depth_size
            depth_t_batch = np.reshape(np.stack(depth_t_batch), (self.batch_size * self.max_steps, shape[0], shape[1], shape[2]))
            action_batch = np.reshape(np.stack(action_batch), (self.batch_size * self.max_steps, self.a_dim))
            reward_batch = np.reshape(np.stack(reward_batch), (self.batch_size * self.max_steps))

            return [depth_t_batch, action_batch, reward_batch, lengths_batch]
        else:
            print 'sample sequences are not enough'
            return None

    def Train(self):
        start_time = time.time()

        batch = self.SampleBatch()

        sample_time =  time.time() - start_time

        if batch is None:
            return
        else:
            depth_t_batch, action_batch, reward_batch, lengths_batch = batch

            #compute target y
            target_a_pred = self.actor.PredictSeqTarget(depth_t_batch, lengths_batch) # b*l, 2
            target_q_pred = self.critic.PredictSeqTarget(depth_t_batch, target_a_pred, lengths_batch) # b*l, 1
            y = []
            for i in xrange(self.batch_size):
                y_seq = np.zeros([self.max_steps])
                for t in xrange(self.max_steps):
                    if t == lengths_batch[i]-1:
                        y_seq[t] = reward_batch[i*self.max_steps+t]
                    elif t < lengths_batch[i]-1:
                        y_seq[t] = reward_batch[i*self.max_steps+t] + self.gamma * target_q_pred[i*self.max_steps+t+1, 0]
                y.append(y_seq)
            y = np.expand_dims(np.stack(y), axis=2)

            y_time = time.time() - start_time - sample_time

            # critic update
            q, _ = self.critic.Train(depth_t_batch, action_batch, y, lengths_batch)

            # actions for a_gradients from critic
            actions = self.actor.PredictSeqOnline(depth_t_batch, lengths_batch)

            # a_gradients
            a_gradients = self.critic.ActionGradients(depth_t_batch, actions, lengths_batch)                                                      

            # actor update
            self.actor.Train(depth_t_batch, lengths_batch, a_gradients[0])

            train_time = time.time() - start_time - sample_time - y_time

            # target networks update
            self.critic.UpdateTarget()
            self.actor.UpdateTarget()

            target_time = time.time() - start_time - sample_time - y_time - train_time

            print 'sample_time:{:.3f}, y_time:{:.3f}, train_time:{:.3f}, target_time:{:.3f}'.format(sample_time,
                                                                                                    y_time,
                                                                                                    train_time,
                                                                                                    target_time)
            
            return q


def main():
    def variable_summaries(var):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    CWD = os.getcwd()

    tf_flags = tf.app.flags

    # network param
    tf_flags.DEFINE_float('a_learning_rate', 1e-3, 'Actor learning rate.')
    tf_flags.DEFINE_float('c_learning_rate', 1e-3, 'Critic learning rate.')
    tf_flags.DEFINE_integer('batch_size', 2, 'Batch size to use during training.')
    tf_flags.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
    tf_flags.DEFINE_integer('n_layers', 1, 'Number of rnn layers in the model.')
    tf_flags.DEFINE_integer('max_steps', 100, 'Max number of steps in an episode.')
    tf_flags.DEFINE_integer('a_dim', 2, 'Dimension of action.')
    tf_flags.DEFINE_integer('depth_h', 128, 'Depth height.')
    tf_flags.DEFINE_integer('depth_w', 160, 'Depth width.')
    tf_flags.DEFINE_integer('depth_c', 3, 'Depth channel.')
    tf_flags.DEFINE_float('a_linear_range', 0.4, 'Range of the linear speed')
    tf_flags.DEFINE_float('a_angular_range', np.pi/4, 'Range of the angular speed')
    tf_flags.DEFINE_float('tau', 0.01, 'Target network update rate')

    # training param
    tf_flags.DEFINE_integer('total_steps', 1000000, 'Total training steps.')
    tf_flags.DEFINE_string('model_dir', os.path.join(CWD, 'saved_network'), 'saved model directory.')
    tf_flags.DEFINE_string('model_name', 'model', 'Name of the model.')
    tf_flags.DEFINE_integer('steps_per_checkpoint', 10000, 'How many training steps to do per checkpoint.')
    tf_flags.DEFINE_integer('buffer_size', 10000, 'The size of Buffer')
    tf_flags.DEFINE_float('gamma', 0.99, 'reward discount')

    # noise param
    tf_flags.DEFINE_float('mu', 0., 'mu')
    tf_flags.DEFINE_float('theta', 0.15, 'theta')
    tf_flags.DEFINE_float('sigma', 0.3, 'sigma')

    flags = tf_flags.FLAGS

    model_dir = os.path.join(flags.model_dir, flags.model_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        agent = RDPG(flags, sess)

        trainable_var = tf.trainable_variables()
        print "  [*] printing trainable variables"
        for idx, v in enumerate(trainable_var):
            print "  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name)
            with tf.name_scope(v.name.replace(':0', '')):
                variable_summaries(v)
        reward_ph = tf.placeholder(tf.float32, [], name='reward')
        q_ph = tf.placeholder(tf.float32, [], name='q_pred')
        tf.summary.scalar('reward', reward_ph)
        tf.summary.scalar('q_estimate', q_ph)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(model_dir, sess.graph)


        sess.run(tf.global_variables_initializer())
        # board_writer = tf.summary.FileWriter('log', sess.graph)
        # board_writer.close()
        q_estimation = []
        for episode in xrange(1, 100):
            print episode
            seq = []
            for t in xrange(0, agent.max_steps):
                seq.append((np.ones([128, 160])*t/np.float(agent.max_steps), [0., 0.], 1./agent.max_steps))
            agent.Add2Mem(seq)

            if episode > agent.batch_size:
                agent.Train()
                # summary = sess.run(merged, feed_dict={reward_ph: 0.,
                #                                       q_ph: 0.})
                # summary_writer.add_summary(summary, episode)
        
            if episode > agent.max_steps:
                q = agent.Train()
                q_estimation.append(q[:agent.max_steps])
            else:
                q_estimation.append(np.zeros([agent.max_steps, 1]))
        q_estimation = np.hstack(q_estimation)
        # print q_estimation
        for t in xrange(agent.max_steps):
            plt.plot(q_estimation[t], label='step{}'.format(t))

        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()