#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import time

class HumanFollower:
    def __init__(self):
        rospy.init_node('human_detection', anonymous=True)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("/meca/diff_drive_controller/cmd_vel", Twist, queue_size=10)

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.human_detected = False
        self.searching = True
        self.reached = False
        rospy.loginfo("Human detection node started.")
        self.rate = rospy.Rate(10)  # 10 Hz

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("CV bridge error: %s", e)
            return

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes, weights = self.hog.detectMultiScale(gray, winStride=(8,8))

        if len(boxes) > 0:
            self.human_detected = True
            self.searching = False
            largest = max(boxes, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest
            center_x = x + w / 2

            # Estimate distance (the larger the height h, the closer the person)
            distance_estimate = 5000.0 / h  # tweak this factor experimentally

            rospy.loginfo("Human detected. Estimated distance: %.2f", distance_estimate)

            # Move backward to human
            twist = Twist()
            twist.linear.x = -0.2  # backward
            move_duration = min(distance_estimate / 0.2, 5.0)  # seconds to move (limit max to 5s)

            start_time = time.time()
            while time.time() - start_time < move_duration and not rospy.is_shutdown():
                self.cmd_vel_pub.publish(twist)
                self.rate.sleep()

            # Stop
            self.cmd_vel_pub.publish(Twist())
            rospy.sleep(1.0)

            # Rotate 180 degrees
            rospy.loginfo("Rotating 180 degrees...")
            twist = Twist()
            twist.angular.z = 0.5  # rad/s
            rotate_time = 3.14 / 0.5  # ~6.28 radians (360°), we use 180° = pi rad

            start_time = time.time()
            while time.time() - start_time < rotate_time and not rospy.is_shutdown():
                self.cmd_vel_pub.publish(twist)
                self.rate.sleep()

            self.cmd_vel_pub.publish(Twist())  # stop
            rospy.loginfo("Done.")
            self.reached = True

        elif not self.human_detected and not self.reached:
            # Still searching
            rospy.loginfo_throttle(5, "Searching for human...")
            twist = Twist()
            twist.angular.z = 0.5  # Rotate in place
            self.cmd_vel_pub.publish(twist)

        # Show output
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Rear Camera - Human Detection", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    try:
        HumanFollower()
    except rospy.ROSInterruptException:
        pass

