import rospy
import math
import time
import numpy as np
import cv2
import copy
import tf
import random
import roslaunch
import pickle

from geometry_msgs.msg import Twist, PoseStamped, Quaternion, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry, Path
from rosgraph_msgs.msg import Clock
from actionlib_msgs.msg import GoalID
from move_base_msgs.msg import MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray
from gazebo_msgs.msg import ModelState, ModelStates

class GazeboWorld():
	def __init__(self, robot_name):
		# initiliaze
		rospy.init_node(robot_name+'_GazeboWorld', anonymous=False, disable_signals=True)

		#------------Params--------------------
		self.depth_image_size = [160, 128]
		self.rgb_image_size = [304, 228]
		self.bridge = CvBridge()
		self.robot_name = robot_name
		self.self_speed = [0.0, 0.0]
		
		self.start_time = time.time()
		self.max_steps = 200
		self.sim_time = Clock().clock
		self.control_freq = 5.


		#-----------Default Robot State-----------------------
		self.default_state = ModelState()
		self.default_state.model_name = robot_name  
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

		rospy.on_shutdown(self.shutdown)

	def ModelStateCallBack(self, data):
		robot_name = copy.deepcopy(self.robot_name)
		if robot_name in data.name:
			idx = data.name.index(robot_name)
			# if data.name[idx] == "mobile_base":
			quaternion = (data.pose[idx].orientation.x,
						  data.pose[idx].orientation.y,
						  data.pose[idx].orientation.z,
						  data.pose[idx].orientation.w)
			euler = tf.transformations.euler_from_quaternion(quaternion)
			self.robot_pose = data.pose[idx]
			self.state_GT = [data.pose[idx].position.x, data.pose[idx].position.y, copy.deepcopy(euler[2])]
			v_x = data.twist[idx].linear.x
			v_y = data.twist[idx].linear.y
			v = np.sqrt(v_x**2 + v_y**2)
			self.speed_GT = [v, data.twist[idx].angular.z]


	def RGBImageCallBack(self, img):
		self.rgb_image = img

	def DepthImageCallBack(self, img):
		self.depth_image = img

	def LaserScanCallBack(self, scan):
		self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
						   scan.scan_time, scan.range_min, scan. range_max]
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
		return copy.deepcopy(self.state_GT)


	def GetSelfSpeedGT(self):
		return copy.deepcopy(self.speed_GT)


	def GetSelfSpeed(self):
		return copy.deepcopy(self.speed)


	def GetSimTime(self):
		return copy.deepcopy(self.sim_time)


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


	def ResetWorld(self):
		self.self_speed = [0.0, 0.0]
		self.start_time = time.time()
		rospy.sleep(0.5)


	def SelfControl(self, action):
		move_cmd = Twist()
		move_cmd.linear.x = action[0]
		move_cmd.linear.y = 0.
		move_cmd.linear.z = 0.
		move_cmd.angular.x = 0.
		move_cmd.angular.y = 0.
		move_cmd.angular.z = action[1]
		self.cmd_vel.publish(move_cmd)


	def shutdown(self):
		rospy.loginfo("Stop Moving")
		self.cmd_vel.publish(Twist())

		rospy.sleep(1)


	def GetRewardAndTerminate(self, t):
		terminate = False
		reset = False
		laser_scan = self.GetLaserObservation()
		laser_min = np.amin(laser_scan)
		[v, w] = self.GetSelfSpeedGT()
		result = 0

		reward = v * np.cos(w) / self.control_freq

		if laser_min < 0.18:
			self.stop_counter += 1
		else:
			self.stop_counter = 0
			
		if self.stop_counter == 2:
			terminate = True
			print 'crash'
			reward = -1.

		return reward, terminate
