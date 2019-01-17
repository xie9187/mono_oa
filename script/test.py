import rospy
import roslaunch
import time

from GazeboWorld import GazeboWorld



def main():
	env = GazeboWorld('robot1')
	rospy.sleep(1.)
	t = 0.
	rate = rospy.Rate(env.control_freq)
	while t < 100 and not rospy.is_shutdown():
		if t % 20 < 10:
			v = 0.4
			w = 0.4
		else:
			v = 0.4
			w = -0.4
		env.SelfControl([v, w])

		t += 1
		rate.sleep()

if __name__ == '__main__':
	main()