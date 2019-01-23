import rospy
import roslaunch
import time
import numpy as np
import csv
import matplotlib.pyplot as plt

from GazeboWorld import GazeboEnv, GazeboRobot


def move_test():
	env = GazeboEnv()
	init_positions = [[-9, 5, 0, 0], [1, 5, 0, 0], [-9, -5, 0, 0], [1, -5, 0, 0]]
	sim_time_start = env.GetSimTime()
	real_time_start = time.time()
	for position_id, init_position in enumerate(init_positions):
		for robot_id in xrange(1,3):
			robot = GazeboRobot('robot'+str(robot_id), 100)
			robot.SetRobotPose(init_position)
			rospy.sleep(1.)

			t = 0.
			rate = rospy.Rate(robot.control_freq)
			with open('log/floor{}_robot{}.csv'.format(position_id, robot_id), 'wb') as csvfile:
				writer = csv.writer(csvfile)
				x = []
				y = []
				while t < 60 and not rospy.is_shutdown():
					if t % 20 < 10:
						v = 0.5
						w = np.pi/4
					else:
						v = 0.5
						w = -np.pi/4
					robot.SelfControl([v, w])
					pose = robot.GetSelfStateGT()
					writer.writerow(pose)
					x.append(pose[0] - init_position[0])
					y.append(pose[1] - init_position[1])
					t += 1
					rate.sleep()
				plt.plot(x, y, label='floor{}-robot{}'.format(position_id, robot_id))
			robot.ResetRobot()
			robot = None
	plt.legend()
	plt.show()
	print "{:.2f} times speed up".format((env.GetSimTime() - sim_time_start) / (time.time() - real_time_start))
	env.CloseGazebo()
	

def object_random_test():
	env = GazeboEnv()
	rospy.sleep(1.)

	object_name_list = env.GetObjectName()
	table = env.RandomiseObject(object_name_list)
	print table

def main():
	# object_random_test()
	move_test()

if __name__ == '__main__':
	main()