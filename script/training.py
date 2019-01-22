import numpy as np
import tensorflow as tf
import math
import os

from model import Actor, Critic
from GazeboWorld import GazeboEnv, GazeboRobot
from ou_noise import OUNoise

CWD = os.getcwd()
cwd_idx = CWD.rfind('/')

tf_flags = tf.app.flags

# network param
tf_flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
tf_flags.DEFINE_integer('batch_size', 32, 'Batch size to use during training.')
tf_flags.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
tf_flags.DEFINE_integer('n_layers', 1, 'Number of rnn layers in the model.')
tf_flags.DEFINE_integer('max_steps', 100, 'Max number of steps in an episode.')
tf_flags.DEFINE_integer('a_dim', 2, 'Dimension of action.')
tf_flags.DEFINE_interger('action_range', [0.4, np.pi/4], 'Range of the actions')
tf_flags.DEFINE_float('tao', 0.01, 'Target network update rate')

# training param
tf_flags.DEFINE_integer('total_steps', 1e6, 'Total training steps.')
tf_flags.DEFINE_string('model_dir', os.path.join(CWD, 'saved_network'), 'saved model directory.')
tf_flags.DEFINE_string('model_name', 'model', 'Name of the model.')
tf_flags.DEFINE_integer('steps_per_checkpoint', 10000, 'How many training steps to do per checkpoint.')
tf_flags.DEFINE_integer('buffer_size', 1e4, 'The size of Buffer')

# noise param
tf_flags.DEFINE_integer('mu', 0., 'mu')
tf_flags.DEFINE_integer('theta', 0.15, 'theta')
tf_flags.DEFINE_integer('sigma', 0.3, 'sigma')

flags = tf_flags.FLAGS

model_dir = os.path.join(flags.model_dir, flags.model_name)
if not gfile.Exists(model_dir): 
    gfile.MkDir(model_dir)

exploration_noise = OUNoise(action_dimension=flags.a_dim, mu=flags.mu, theta=flags.theta, sigma=flags.sigma)


def main():
	env = GazeboEnv()
	env.LaunchGazebo()
	object_name_list = env.GetObjectName()
	table = env.RandomiseObject(object_name_list)
	robot = GazeboRobot('robot'+str(robot_id))


if __name__ == '__main__':
    tf.app.run() 