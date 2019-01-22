import tensorflow as tf
import numpy as np
import model_utils as model_utils
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
        b_a = tf.get_variable('b_a', [2], initializer=tf.initializers.random_uniform(-0.003, 0.003))

        # training
        depth_vectors = tf.reshape(conv3, (self.batch_size, self.max_steps, shape[1]*shape[2]*shape[3])) # b, l, h

        rnn_outputs, _ = tf.nn.dynamic_rnn(rnn_cell, 
                                            depth_vectors, 
                                            sequence_length=self.lengths,
                                            dtype=tf.float32) # b, l, h

        rnn_outputs_reshape = tf.reshape(rnn_outputs, [-1, self.n_hidden]) # b*l, h

        a_linear = tf.nn.tanh(tf.matmul(rnn_outputs_reshape, w_linear_a)) * self.action_range[0] # b*l, 1
        a_angular = tf.nn.sigmoid(tf.matmul(rnn_outputs_reshape, w_angular_a)) * self.action_range[1] # b*l, 1
        a = tf.concat([a_linear, a_angular], axis=1) + b_a # b*l, 2
        # a = tf.reshape(a, (self.batch_size, self.max_steps, 2))

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

        a_linear_test = tf.nn.tanh(tf.matmul(rnn_outputs_test, w_linear_a)) * self.action_range[0] # b*l, 1
        a_angular_test = tf.nn.sigmoid(tf.matmul(rnn_outputs_test, w_angular_a)) * self.action_range[1] # b*l, 1
        a_test = tf.concat([a_linear_test, a_angular_test], axis=1) + b_a # b*l, 2

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

        self.loss_t = tf.reduce_sum(self.square_diff, reduction_indices=1) / tf.cast(self.lengths, tf.float32) # b, 1
        self.loss_n = tf.reduce_sum(self.loss_t, reduction_indices=0) / self.batch_size # 1

        self.gradient = tf.gradients(self.loss_n, self.network_params)
        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.opt.apply_gradients(zip(self.gradient, self.network_params))

        self.action_grads = tf.gradients(self.q_online, self.action_input)

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
        self.depth_size = flags
        self.n_hidden = flags.n_hidden
        self.learning_rate = flags.learning_rate
        self.batch_size = flags.batch_size
        self.max_steps = flags.max_steps
        self.tau = flags.tau
        self.n_layers = flags.n_layers
        self.action_range = action_range
        self.buffer_size = flags.buffer_size

        self.actor = Actor(sess=sess,
                           depth_size=self.depth_size,
                           n_hidden=self.n_hidden,
                           max_steps=self.max_steps,
                           learning_rate=self.learning_rate,
                           batch_size=self.batch_size,
                           action_range=self.action_range,
                           tau=self.tau,
                           n_layers=self.n_layers)
        self.critic = Critic(sess=sess,
                             depth_size=self.depth_size,
                             n_hidden=self.n_hidden,
                             max_steps=self.max_steps,
                             learning_rate=self.learning_rate,
                             batch_size=self.batch_size,
                             num_actor_vars=len(self.actor.network_params)+len(self.actor.target_network_params),
                             tau=self.tau,
                             n_layers=self.n_layers)
        self.memory = []

    def ActorPredict(self, depth_input, prev_rnn_state_online=(np.zeros([1, self.n_hidden]), np.zeros([1, self.n_hidden]))):
        return self.actor.Predict(depth_input, prev_rnn_state_online)

    def Add2Mem(self, seq):
        self.memory.append(seq) #seq: ((d_0, a_0, r_0), (d_1, a_1, r_1), ...)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def SampleBatch(self):
        if len(self.memory) > self.batch_size:
            indices = np.random.randint(0, len(self.memory), size=(self.batch_size))
            for idx in indices:
                sampled_seq = memory[idx]
                depth_full_seq = 
                for seq_idx in xrange(self.batch_size):
                    pass
                depth_input_seq = np.zeros()
        else:
            return None


def main():
    depth_size=(128, 160, 1)
    n_hidden=128
    max_steps=100
    learning_rate=1e-4
    batch_size=16
    action_range=[0.4, np.pi/4]
    tau=0.01
    n_layers=1
    with tf.Session() as sess:
        actor = Actor(sess=sess,
                      depth_size=depth_size,
                      n_hidden=n_hidden,
                      max_steps=max_steps,
                      learning_rate=learning_rate,
                      batch_size=batch_size,
                      action_range=action_range,
                      tau=tau,
                      n_layers=n_layers)
        # for var in actor.network_params:
        #     print var

        critic = Critic(sess=sess,
                      depth_size=depth_size,
                      n_hidden=n_hidden,
                      max_steps=max_steps,
                      learning_rate=learning_rate,
                      batch_size=batch_size,
                      num_actor_vars=len(actor.network_params)+len(actor.target_network_params),
                      tau=tau,
                      n_layers=n_layers)
        # for var in critic.network_params:
        #     print var

        depth_input = np.random.rand(batch_size * max_steps, 128, 160, 1)
        action_input = np.random.rand(batch_size * max_steps, 2)
        depth_input_test = np.random.rand(1, 128, 160, 1)
        action_input_test = np.random.rand(1, 2)
        # lengths = np.random.randint(2, max_steps, size=(16))
        lengths = np.ones([16]) * max_steps
        predicted_q = np.random.rand(batch_size, max_steps, 1)
        prev_rnn_state_online = (np.zeros([1, 128]), np.zeros([1, 128]))

        sess.run(tf.global_variables_initializer())
        # board_writer = tf.summary.FileWriter('log', sess.graph)
        # board_writer.close()

        q, _ = critic.Train(depth_input, action_input, predicted_q, lengths)
        print 'q', q
        print 'seq q online', critic.PredictSeqOnline(depth_input, action_input, lengths)
        print 'seq q target', critic.PredictSeqTarget(depth_input, action_input, lengths)
        print 'q online', critic.Predict(depth_input_test, action_input_test, prev_rnn_state_online)
        print 'a_gradient', critic.ActionGradients(depth_input, action_input, lengths)[0]
        critic.UpdateTarget()

        a_gradient = critic.ActionGradients(depth_input, action_input, lengths)[0]

        actor.Train(depth_input, lengths, a_gradient)
        print actor.PredictSeqTarget(depth_input, lengths)
        print actor.Predict(depth_input_test, prev_rnn_state_online)
        actor.UpdateTarget()

if __name__ == '__main__':
    main()