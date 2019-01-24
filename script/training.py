import numpy as np
import tensorflow as tf
import math
import os
import rospy
import time 

from rdpg import RDPG
from GazeboWorld import GazeboEnv, GazeboRobot
from ou_noise import OUNoise

CWD = os.getcwd()

tf_flags = tf.app.flags

# network param
tf_flags.DEFINE_float('a_learning_rate', 1e-3, 'Actor learning rate.')
tf_flags.DEFINE_float('c_learning_rate', 1e-3, 'Critic learning rate.')
tf_flags.DEFINE_integer('batch_size', 16, 'Batch size to use during training.')
tf_flags.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
tf_flags.DEFINE_integer('n_layers', 1, 'Number of rnn layers in the model.')
tf_flags.DEFINE_integer('max_steps', 100, 'Max number of steps in an episode.')
tf_flags.DEFINE_integer('a_dim', 2, 'Dimension of action.')
tf_flags.DEFINE_integer('depth_h', 128, 'Depth height.')
tf_flags.DEFINE_integer('depth_w', 160, 'Depth width.')
tf_flags.DEFINE_integer('depth_c', 1, 'Depth channel.')
tf_flags.DEFINE_float('a_linear_range', 0.4, 'Range of the linear speed')
tf_flags.DEFINE_float('a_angular_range', np.pi/4, 'Range of the angular speed')
tf_flags.DEFINE_float('tau', 0.01, 'Target network update rate')

# training param
tf_flags.DEFINE_integer('total_steps', 1000000, 'Total training steps.')
tf_flags.DEFINE_string('model_dir', os.path.join(CWD, 'saved_network'), 'saved model directory.')
tf_flags.DEFINE_string('model_name', 'model', 'Name of the model.')
tf_flags.DEFINE_integer('steps_per_checkpoint', 10000, 'How many training steps to do per checkpoint.')
tf_flags.DEFINE_integer('buffer_size', 1000, 'The size of Buffer')
tf_flags.DEFINE_float('gamma', 0.99, 'reward discount')

# noise param
tf_flags.DEFINE_float('mu', 0., 'mu')
tf_flags.DEFINE_float('theta', 0.15, 'theta')
tf_flags.DEFINE_float('sigma', 0.2, 'sigma')

flags = tf_flags.FLAGS

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

def main():
    env = GazeboEnv()
    object_name_list = env.GetObjectName()
    map_table = env.RandomiseObject(object_name_list)
    robot = GazeboRobot('robot1', flags.max_steps)
    rate = rospy.Rate(robot.control_freq)

    exploration_noise = OUNoise(action_dimension=flags.a_dim, 
                                mu=flags.mu, theta=flags.theta, sigma=flags.sigma)
   
    with tf.Session() as sess:
        agent = RDPG(flags, sess)

        trainable_var = tf.trainable_variables()

        model_dir = os.path.join(flags.model_dir, flags.model_name)
        if not os.path.exists(model_dir): 
            os.makedirs(model_dir)

        # summary
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

        # model saver
        saver = tf.train.Saver(trainable_var, max_to_keep=3)
        
        sess.run(tf.global_variables_initializer())

        # start training
        T = 0
        episode = 0
        while T < flags.total_steps:
            t = 0
            init_pose = robot.SetInitialPose(map_table)
            seq = []
            total_reward = 0
            loop_time = []
            while not rospy.is_shutdown():
                start_time = time.time()
                reward, terminate = robot.GetRewardAndTerminate(t)
                if t > 0:
                    seq.append((depth_img, action, reward))

                depth_img = np.reshape(robot.GetDepthImageObservation(), agent.depth_size)
                action = agent.ActorPredict([depth_img], t)[0] \
                         + exploration_noise.noise() \
                         * np.asarray(agent.action_range)

                robot.SelfControl(action, agent.action_range)

                total_reward += reward

                if (T + 1) % flags.steps_per_checkpoint == 0:
                    saver.save(sess, os.path.join(model_dir, 'network') , global_step=T)

                if terminate:
                    if len(seq) > 10:
                        agent.Add2Mem(seq)
                    if episode > agent.batch_size:
                        for train_step in xrange(2):
                            q = agent.Train()
                        summary = sess.run(merged, feed_dict={reward_ph: total_reward,
                                                              q_ph: np.amax(q)})
                        summary_writer.add_summary(summary, episode)
                        print 'Episode:{:} | Steps:{:} | Reward:{:.2f} | T:{:} | Q:{:.2f}'.format(episode, 
                                                                                                  t, 
                                                                                                  total_reward, 
                                                                                                  T, 
                                                                                                  np.amax(q))
                    else: 
                        print 'Episode:{:} | Steps:{:} | Reward:{:.2f} | T:{:}'.format(episode, 
                                                                                       t, 
                                                                                       total_reward, 
                                                                                       T)
                    episode +=1
                    T += 1
                    break

                t += 1
                T += 1
                rate.sleep()

            




if __name__ == '__main__':
    main()