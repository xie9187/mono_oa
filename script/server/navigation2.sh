#!/bin/bash

export ROS_IP=192.168.0.118
export ROS_MASTER_URI=http://192.168.0.118:11313
export GAZEBO_MASTER_URI=http://192.168.0.118:11353

DISPLAY=:8 vglrun -d :7.3 python script/training --model_name=ddpg
