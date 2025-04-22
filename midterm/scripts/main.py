#!/usr/bin/env python3
from utils import utils
import rospy

if __name__ == "__main__":
    try:
        rospy.loginfo("This project belongs to Thanh Tuan AKA Ein Schöner Mann...")
        controller = utils.Controller()
        controller.move()
    except rospy.ROSInterruptException:
        pass
