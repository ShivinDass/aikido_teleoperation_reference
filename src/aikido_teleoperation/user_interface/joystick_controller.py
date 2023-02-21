import numpy as np
from sensor_msgs.msg import Joy
import rospy
from threading import Lock


class JoystickController:
    def __init__(self):
        self.inputlock = Lock()

        self.scale_actions = 0.2
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
        data = np.concatenate((np.array(message.axes)*self.scale_actions, [0,0]))
        data[1] *= -1
        return data

if __name__=='__main__':
    rospy.init_node('cam_node', anonymous=True)
    a = JoystickController()
    # rospy.spin()
    print(a.get_action())