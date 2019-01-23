import rospy
import math
import time
import numpy as np
import cv2
import copy
import tf
import random
import pickle
import roslaunch
import os 

from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from actionlib_msgs.msg import GoalID
from gazebo_msgs.msg import ModelState, ModelStates
from scipy.ndimage import uniform_filter

PWD = os.getcwd()

class GazeboEnv():
	def __init__(self):
		# initiliaze
		# self.LaunchGazebo()
		rospy.init_node('GazeboWorld', anonymous=False, disable_signals=True)

		#------------Params--------------------
		self.world_start_time = time.time()
		self.sim_time = Clock().clock
		self.model_names = None
		self.model_poses = None

		#-----------Default Robot State-----------------------
		self.default_state = ModelState()
		self.default_state.model_name = None  
		self.default_state.pose.position.x = 0.
		self.default_state.pose.position.y = 0.
		self.default_state.pose.position.z = 0.
		self.default_state.pose.orientation.x = 0.0
		self.default_state.pose.orientation.y = 0.0
		self.default_state.pose.orientation.z = 0.0
		self.default_state.pose.orientation.w = 1.0
		self.default_state.twist.linear.x = 0.
		self.default_state.twist.linear.y = 0.
		self.default_state.twist.linear.z = 0.
		self.default_state.twist.angular.x = 0.
		self.default_state.twist.angular.y = 0.
		self.default_state.twist.angular.z = 0.
		self.default_state.reference_frame = 'world'

		#-----------Publisher and Subscriber-------------
		self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 100)

		self.model_state_sub = rospy.Subscriber('gazebo/model_states', ModelStates, self.ModelStateCallBack)
		self.sim_clock = rospy.Subscriber('clock', Clock, self.SimClockCallBack)

		while self.model_names is None and not rospy.is_shutdown():
			pass
		print "Env intialised"


	def ModelStateCallBack(self, data):
		self.model_names = data.name
		self.model_poses = data.pose


	def SimClockCallBack(self, clock):
		self.sim_time = clock.clock.secs + clock.clock.nsecs/1e9


	def GetSimTime(self):
		return copy.deepcopy(self.sim_time)


	def GetObjectName(self):
		name_list = []
		back_list = ['Floor', 'env', 'robot', 'ground', 'door']
		for name in self.model_names:
			append_flag = True
			for black_name in back_list:
				if black_name in name:
					append_flag = False
			if append_flag:
				name_list.append(name)
		return name_list


	def RandomiseObject(self, name_list):
		table = np.zeros([20, 20], dtype='uint8')
		table[18:, 9:11] = 1
		table[7:10, 8:12] = 1
		empty_space = np.stack(np.where(table==0), axis=1)
		object_table_pos_idx = random.sample(range(0, len(empty_space)), len(name_list))
		large_object = ['bookshelf_large', 'desk', 'sofa_set']
		for name, idx in zip(name_list, object_table_pos_idx):
			[y, x] = empty_space[idx]
			table[y, x] = 1
			[y_real, x_real] = [9.5 - y, x - 9.5]
			theta_real = 0.
			large_flag = False
			for large_obj_name in large_object:
				if large_obj_name in name:
					large_flag = True
			if large_flag:
				if np.random.random() < 0.5:
					table[y, np.amax([0, x - 1]) : np.amin([20, x + 2])] = 1
				else:
					theta_real = np.pi/2
					table[np.amax([0, y - 1]) : np.amin([20, y + 2]), x] = 1
			self.SetObjectPose(name, [x_real, y_real, 0.,theta_real])
		return table


	def SetObjectPose(self, name, pose):
		object_state = copy.deepcopy(self.default_state)
		object_state.model_name = name
		object_state.pose.position.x = pose[0]
		object_state.pose.position.y = pose[1]
		object_state.pose.position.z = pose[2]
		quaternion = tf.transformations.quaternion_from_euler(0., 0., pose[3])
		object_state.pose.orientation.x = quaternion[0]
		object_state.pose.orientation.y = quaternion[1]
		object_state.pose.orientation.z = quaternion[2]
		object_state.pose.orientation.w = quaternion[3]

		self.set_state.publish(object_state)
		start_time = time.time()
		while time.time() - start_time < 0.5 and not rospy.is_shutdown():
			self.set_state.publish(object_state)
		print 'Set ' + name


	def LaunchGazebo(self):
		# self.RandomiseRobotMass()

		uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
		roslaunch.configure_logging(uuid)
		self.launch = roslaunch.parent.ROSLaunchParent(uuid, [os.path.join(PWD, "launch/gazebo_world.launch")])
		self.launch.start()
		rospy.loginfo("started")
		time.sleep(30)
		print 'gazebo launched'

	def CloseGazebo(self):
		self.launch.shutdown()
		# time.sleep(30)
		print 'gazebo closed'

	def RandomiseRobotMass(self):
		m = np.random.random()*50
		h = 0.09
		r = 0.175

		ixx = m/12. * (3 * r**2 + h**2)
		iyy = m/12. * (3 * r**2 + h**2)
		izz = 0.5 * m * r**2

		file_names = ['./robots/launch/kobuki_hexagons_hokuyo1.xml',
					  './robots/launch/kobuki_hexagons_hokuyo2.xml']
		for file_name in file_names:
			with open(file_name, 'r') as file:
				file = launch_file.readlines()
				file[67] = '      <mass value="{:.6f}"/>\n'.format(m)
				file[76] = '      <inertia ixx="{:.6f}" ixy="0.0" ixz="0.0" iyy="{:.6f}" iyz="0.0" izz="{:.6f}"/>\n'.format(ixx, iyy, izz)
			with open(file_name, 'w') as launch_file:
				launch_file.writelines(file)

class GazeboRobot():
	def __init__(self, robot_name, max_steps):
		#------------Params--------------------
		self.depth_image_size = [160, 128]
		self.rgb_image_size = [304, 228]
		self.bridge = CvBridge()
		self.robot_name = robot_name
		self.self_speed = [0.0, 0.0]
		self.start_time = time.time()
		self.max_steps = max_steps
		self.sim_time = Clock().clock
		self.control_freq = 5.
		self.pose = None
		default_positions = {'robot1': [11., 0., 0.],
							 'robot2': [11., 1., 0.],
							 'robot3': [11., 2., 0.],
							 'robot4': [11., 3., 0.],
							 'robot5': [11., 4., 0.]}

		#-----------Default Robot State-----------------------
		self.default_state = ModelState()
		self.default_state.model_name = robot_name  
		self.default_state.pose.position.x = default_positions[robot_name][0]
		self.default_state.pose.position.y = default_positions[robot_name][1]
		self.default_state.pose.position.z = default_positions[robot_name][2]
		self.default_state.pose.orientation.x = 0.0
		self.default_state.pose.orientation.y = 0.0
		self.default_state.pose.orientation.z = 0.0
		self.default_state.pose.orientation.w = 1.0
		self.default_state.twist.linear.x = 0.
		self.default_state.twist.linear.y = 0.
		self.default_state.twist.linear.z = 0.
		self.default_state.twist.angular.x = 0.
		self.default_state.twist.angular.y = 0.
		self.default_state.twist.angular.z = 0.
		self.default_state.reference_frame = 'world'

		#-----------Publisher and Subscriber-------------
		self.cmd_vel = rospy.Publisher(robot_name+'/mobile_base/commands/velocity', Twist, queue_size = 10)
		self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 100)
		self.resized_depth_img = rospy.Publisher(robot_name+'/camera/depth/image_resized',Image, queue_size = 10)
		self.resized_rgb_img = rospy.Publisher(robot_name+'/camera/rgb/image_resized',Image, queue_size = 10)

		self.object_state_sub = rospy.Subscriber('gazebo/model_states', ModelStates, self.ModelStateCallBack)
		self.odom_sub = rospy.Subscriber(robot_name+'/odom', Odometry, self.OdometryCallBack)
		self.sim_clock = rospy.Subscriber('clock', Clock, self.SimClockCallBack)
		self.depth_image_sub = rospy.Subscriber(robot_name+'/camera/depth/image_raw', Image, self.DepthImageCallBack)
		self.rgb_image_sub = rospy.Subscriber(robot_name+'/camera/rgb/image_raw', Image, self.RGBImageCallBack)
		self.laser_sub = rospy.Subscriber('mybot/laser/scan', LaserScan, self.LaserScanCallBack)


	def ModelStateCallBack(self, data):
		idx = data.name.index(self.robot_name)
		self.pose = data.pose[idx]
		self.twist = data.twist[idx]

	def RGBImageCallBack(self, img):
		self.rgb_image = img

	def DepthImageCallBack(self, img):
		self.depth_image = img

	def LaserScanCallBack(self, scan):
		if self.robot_name in scan.header.frame_id:
			self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
							   scan.scan_time, scan.range_min, scan.range_max]
			self.scan = np.array(scan.ranges)

	def OdometryCallBack(self, odometry):
		Quaternions = odometry.pose.pose.orientation
		Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
		self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
		self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

	def SimClockCallBack(self, clock):
		self.sim_time = clock.clock


	def GetDepthImageObservation(self):
		# ros image to cv2 image

		try:
			cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, "32FC1")
		except Exception as e:
			raise e

		cv_img = np.array(cv_img, dtype=np.float32)
		# resize
		dim = (self.depth_image_size[0], self.depth_image_size[1])
		cv_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)

		cv_img[np.isnan(cv_img)] = 0.
		cv_img[cv_img < 0.4] = 0.
		cv_img/=(10./255.)

		# # inpainting
		# mask = copy.deepcopy(cv_img)
		# mask[mask == 0.] = 1.
		# mask[mask != 1.] = 0.
		# mask = np.uint8(mask)
		# cv_img = cv2.inpaint(np.uint8(cv_img), mask, 3, cv2.INPAINT_TELEA)

		cv_img = np.array(cv_img, dtype=np.float32)
		cv_img*=(10./255.)

		# cv2 image to ros image and publish
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
		except Exception as e:
			raise e
		self.resized_depth_img.publish(resized_img)
		return(cv_img/5.)


	def GetRGBImageObservation(self):
		# ros image to cv2 image
		try:
			cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
		except Exception as e:
			raise e
		# resize
		dim = (self.rgb_image_size[0], self.rgb_image_size[1])
		cv_resized_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)
		# cv2 image to ros image and publish
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
		except Exception as e:
			raise e
		self.resized_rgb_img.publish(resized_img)
		return(cv_resized_img)


	def GetLaserObservation(self):
		scan = copy.deepcopy(self.scan)
		scan[np.isnan(scan)] = 5.6
		scan[np.isinf(scan)] = 5.6
		return scan


	def GetSelfStateGT(self):
		quaternion = (self.pose.orientation.x,
					  self.pose.orientation.y,
					  self.pose.orientation.z,
					  self.pose.orientation.w)
		euler = tf.transformations.euler_from_quaternion(quaternion)
		self.state_GT = [self.pose.position.x, self.pose.position.y, euler[2]]
		return copy.deepcopy(self.state_GT)


	def GetSelfSpeedGT(self):
		v_x = self.twist.linear.x
		v_y = self.twist.linear.y
		v = np.sqrt(v_x**2 + v_y**2)
		self.speed_GT = [v, self.twist.angular.z]
		return copy.deepcopy(self.speed_GT)

	def GetSelfState(self):
		return copy.deepcopy(self.state)


	def GetSelfSpeed(self):
		return copy.deepcopy(self.speed)


	def GetSimTime(self):
		return copy.deepcopy(self.sim_time)


	def SetRobotPose(self, pose=None):
		object_state = copy.deepcopy(self.default_state)
		object_state.model_name = self.robot_name
		if pose is not None:
			object_state.pose.position.x = pose[0]
			object_state.pose.position.y = pose[1]
			object_state.pose.position.z = pose[2]
			quaternion = tf.transformations.quaternion_from_euler(0., 0., pose[3])
			object_state.pose.orientation.x = quaternion[0]
			object_state.pose.orientation.y = quaternion[1]
			object_state.pose.orientation.z = quaternion[2]
			object_state.pose.orientation.w = quaternion[3]

		self.set_state.publish(object_state)
		start_time = time.time()
		while time.time() - start_time < 0.5 and not rospy.is_shutdown():
			self.set_state.publish(object_state)


	def ResetRobot(self):
		self.SetRobotPose()


	def SelfControl(self, action, action_range):

		if action[0] < 0.:
			action[0] = 0.
		if action[0] > action_range[0]:
			action[0] = action_range[0]
		if action[1] < -action_range[1]:
			action[1] = -action_range[1]
		if action[1] > action_range[1]:
			action[1] = action_range[1]

		move_cmd = Twist()
		move_cmd.linear.x = action[0]
		move_cmd.linear.y = 0.
		move_cmd.linear.z = 0.
		move_cmd.angular.x = 0.
		move_cmd.angular.y = 0.
		move_cmd.angular.z = action[1]
		self.cmd_vel.publish(move_cmd)


	def GetRewardAndTerminate(self, t):
		terminate = False
		laser_scan = self.GetLaserObservation()
		laser_min = np.amin(laser_scan)
		[v, w] = self.GetSelfSpeedGT()
		result = 0

		reward = v * np.cos(w) / self.control_freq

		if laser_min < 0.2:
		# 	self.stop_counter += 1
		# else:
		# 	self.stop_counter = 0
			
		# if self.stop_counter == 2:
			terminate = True
			reward = -1.

		if t >= self.max_steps - 1:
			terminate = True

		return reward, terminate


	def SetInitialPose(self, table):
		# mean_table = uniform_filter(np.asarray(table, dtype='float32'), size=5, mode='constant', cval=1.)
		# positions = np.stack(np.where(mean_table == np.amin(mean_table)), axis=1)

		# for i in xrange(0,100):
		# 	y, x = random.sample(positions, 1)[0]
		# 	if table[y, x] == 0:
		# 		break
		# if i == 99:

		positions = np.stack(np.where(table == 0), axis=1)
		y, x = random.sample(positions, 1)[0]
		y_real = 9.5 - y
		x_real = x - 9.5
		theta_real = (np.random.random() - 0.5) * np.pi * 2

		self.SetRobotPose([x_real, y_real, 0, theta_real])

