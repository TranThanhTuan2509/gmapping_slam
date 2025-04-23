#!/bin/bash

echo "🔁 Đang điều khiển tay máy... (nhấn Ctrl + C để dừng)"

while true; do
  echo "👉 arm1 = 1.57 | arm2 = 1.57"
  rostopic pub -1 /arm1_joint_position_controller/command std_msgs/Float64 "data: 1.57"
  rostopic pub -1 /arm2_joint_position_controller/command std_msgs/Float64 "data: 1.57"
  sleep 3

  echo "👈 arm1 = -1.57 | arm2 = -0.8"
  rostopic pub -1 /arm1_joint_position_controller/command std_msgs/Float64 "data: -1.57"
  rostopic pub -1 /arm2_joint_position_controller/command std_msgs/Float64 "data: -0.8"
  sleep 3
done

