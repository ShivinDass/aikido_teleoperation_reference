from datetime import date
import numpy as np
from sensor_msgs.msg import Joy
import rospy
from threading import Lock

class DS4Input:

    def __init__(self, message, control_mode,scale_actions, finger_speed):
        if control_mode=='gripper':
            self.axes = np.array([message.axes[0], -message.axes[1], message.axes[4]])*scale_actions # wrt gripper cam
        elif control_mode=='3pCam':
            self.axes = np.array([-message.axes[0], message.axes[1], message.axes[4]])*scale_actions # wrt 3rd person cam
        else:
            self.axes = np.array([-message.axes[1], -message.axes[0], message.axes[4]])*scale_actions

        self.override_control = message.axes[5] < 0
        self.display_imgs = message.axes[2] < -0.9

        if message.buttons[5]==1:
            self.gripper = np.ones(2)
        elif message.buttons[4]==1:
            self.gripper = -1*np.ones(2)
        else:
            self.gripper = np.zeros(2)

        self.action = np.concatenate((self.axes, self.gripper*finger_speed))
        self.done = message.buttons[1]

    def __str__(self):
        return "Axes: " + ' '.join(map(str, self.axes)) + "\nGripper: " + ' '.join(map(str, self.gripper)) + \
                    "\Done: " + str(self.done) + "\nOverride:" + str(self.override_control) + "\nDisplay_imgs:" + str(self.display_imgs)

class DS4Controller:
    def __init__(self, control_mode='real', finger_speed = 0.3):
        self.inputlock = Lock()

        self.finger_speed = finger_speed
        self.scale_actions = 0.2
        self.control_mode = control_mode

        self.input_topic_name = "/joy"
        self.input_message_type = Joy

        self.most_recent_message = None
        self.init_listener()

    def callback(self, data):
        with self.inputlock:
            self.most_recent_message = data

    def init_listener(self):
        rospy.Subscriber(self.input_topic_name, self.input_message_type, self.callback)

    def get_action(self):
        while self.most_recent_message is None:
            print('Waiting for topic {} to publish.'.format(self.input_topic_name))
            rospy.sleep(0.02)
        with self.inputlock:
            data = self.message_to_data(self.most_recent_message)

        return data
    
    def message_to_data(self, message):
        data = DS4Input(message, control_mode=self.control_mode, scale_actions=self.scale_actions, finger_speed = self.finger_speed)
        return data

if __name__=='__main__':
    rospy.init_node('cam_node', anonymous=True)
    a = DS4Controller()
    import time
    for _ in range(50):
        print(a.get_action())
        time.sleep(0.1)