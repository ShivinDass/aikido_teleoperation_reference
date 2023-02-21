import time
from threading import Lock
from collections import OrderedDict

import gym
import numpy as np
import rospy
import adapy

from servo_controller import ServoController

from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image, CompressedImage

from controllers.jaco_ik_controller import JacoIKController
import controllers.transform_utils as T

def Initialize_Adapy():
    """ Initializes robot and environment through adapy, using the specified environment path
    @param env_path path to OpenRAVE environment
    @return environment, robot
    """
    ada = adapy.Ada(False)
    viewer = ada.start_viewer('dart_markers/simple_trajectories', 'map')
    world = ada.get_world()
    rospy.sleep(1.0)
    # init_robot(ada)
    return ada, viewer, world

# def init_robot(robot):
#     robot.set_arm_positions([4.8, 2.9147, 1.009, 4.1957, 1.44237, 1.3166])

class JacoGym(gym.Env):

    def __init__(self, visual_ob=False, frequency=10):
        super(JacoGym).__init__()

        self.frequency = frequency
        self.visual_ob =visual_ob

        robot, viewer, world = Initialize_Adapy()

        self.world = world
        self.robot = robot
        self.hand = robot.get_hand()

        self.min_pos_lim = [-0.45, -0.67, 0.15] # min values for cartesian poses(define in simulation poses)
        self.max_pos_lim = [0.22, -0.19, 0.40]  # max values for cartesian poses (define in simulation poses)
        self.max_speed_limit_weight = 1       # ratio to limit the max speed of the arm by
        self._action_repeat = 30                 # number of times the ik is calculated and executed in each step cycle
        self.step_size = 0.02                   # step size defining the speed of the robot

        # construct servo controller
        self.servo_controller = ServoController(False, 1 / 10, 0.3, self.robot)

        if visual_ob:
            self.front_camera_listener = CameraListener(input_topic_name="camera/image_raw", input_message_type=Image, message_to_data=webcam_postprocessing)
            self.mount_camera_listener = CameraListener(input_topic_name="realsense/color/image_raw", input_message_type=Image, message_to_data=realsense_postprocessing)

    @property
    def observation_space(self):
        ob_space = OrderedDict()

        img_dim = 224
        if self.visual_ob:
            ob_space["front_cam_ob"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(img_dim, img_dim, 3),
                dtype=np.uint8
            )

            ob_space["mount_cam_ob"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(img_dim, img_dim, 3),
                dtype=np.uint8
            )

        ob_space["joint_pos_ob"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6+2,),
        )

        ob_space["joint_vel_ob"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6+2,),
        )

        ob_space["ee_cartesian_pos_ob"] = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(3+4,)
        )

        ob_space["ee_cartesian_vel_ob"] = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(3+3,)
        )

        return gym.spaces.Dict(ob_space)
    
    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-1,
            high=1,
            shape=(5,),
            dtype=np.float32
        )

    def _get_latest_cam(self):
        images = {
            'front_cam_ob': self.front_camera_listener.get_most_recent_msg(),
            'mount_cam_ob': self.mount_camera_listener.get_most_recent_msg()
        }

        return images
    
    def _reset_to_home(self):
        self.stabilize_jaco()

        # manual resetting for now so skip
    
    def _get_obs(self, twist_vel=np.zeros(6), finger_vel=np.zeros(2)):
        state = OrderedDict()
        if self.visual_ob:
            cams = self._get_latest_cam()
            state["front_cam_ob"] = cams['front_cam_ob']
            state["mount_cam_ob"] = cams['mount_cam_ob']

        state["joint_pos_ob"] = np.concatenate((self.robot.get_arm_positions(), self.hand.get_positions()), axis=0)
        state["joint_vel_ob"] = np.concatenate((self.robot.compute_joint_velocity_from_twist(twist_vel, self.step_size), finger_vel), axis=0)
        
        ee_pos, ee_rot, ee_pos_vel, ee_rot_vel = self._controller.get_cartesian_info(self.robot.compute_joint_velocity_from_twist(twist_vel, self.step_size))
        state["ee_cartesian_pos_ob"] = np.concatenate((ee_pos, ee_rot), axis=0)
        state["ee_cartesian_vel_ob"] = np.concatenate((ee_pos_vel, ee_rot_vel), axis=0)
        return state

    def _robot_jpos_getter(self):
        return self.robot.get_arm_positions()
    
    def _hand_jpos_getter(self):
        return self.hand.get_positions()

    def reset(self):
        self._reset_to_home()

        print("ALERT: RESET THE ENVIRONMENT OBJECTS BACK TO THEIR POSITIONS!")
        input("Press Enter when reset is complete:")
        
        self._controller = JacoIKController(
            bullet_data_path="/root/code/teleop/aikido_teleoperation/src/",
            robot_jpos_getter=self._robot_jpos_getter,
            hand_jpos_getter=self._hand_jpos_getter,
            min_pos_lim=self.min_pos_lim,
            max_pos_lim=self.max_pos_lim
        )

        # measuring time to maintain frequency
        self.start_time = None
        self.end_time = None

        return self._get_obs()

    def step(self, action):

        pos_vel = action[:3]
        finger_vel = action[3:]

        twist_vel = np.zeros(6)
        twist_vel[3:] = pos_vel

        self._execute_ik_twist(twist_vel)
        self._execute_finger_velocities(finger_vel)

        self.end_time = time.time()
        if self.start_time is not None:
            print('Idle time:', max(0., 1/self.frequency - (self.end_time - self.start_time)))
            rospy.sleep(max(0., 1/self.frequency - (self.end_time - self.start_time)))

        self.start_time = time.time()

        obs = self._get_obs(twist_vel, finger_vel)
        rew = 0
        done = False
        info = {}

        return obs, rew, done, info
    
    def _execute_ik_twist(self, twist): 
        '''
            twist[:3] -> orientation
            twist[3:] -> pos 
        '''
        d_pos = twist[3:] * self.step_size * 5 #rescaling for easy movement
        d_pos = np.clip([d_pos[0], d_pos[1], d_pos[2]], -0.03, 0.03) #clip d_pos to limit max cartesian velocity
        input = {
                    "dpos": np.array(d_pos),
                    "rotation": T.quat2mat([0,0,0,1])
                }
        velocities = self._controller.get_control(**input)

        # keep trying to reach the target in a closed-loop
        velocities = np.clip(velocities, -1, 1)
        for i in range(self._action_repeat):
            self._execute_joint_velocities(joint_vels=velocities)
            
            if i + 1 < self._action_repeat:
                velocities = np.clip(self._controller.get_control(), -1, 1)

    def _execute_twist(self, twist):
        velocities = self.robot.compute_joint_velocity_from_twist(twist, self.step_size)
        self._execute_joint_velocities(velocities)

    def _execute_joint_velocities(self, joint_vels):
        joint_vel_limits = self.max_speed_limit_weight*self.robot.get_arm_velocity_limits()
        ratio = np.absolute(joint_vels / joint_vel_limits)
        if np.max(ratio) > 0.95:
            joint_vels /= np.max(ratio) / 0.95

        self.servo_controller.servo_arm(joint_vels, self.robot)

    def _execute_finger_velocities(self, finger_velocites):
        self.servo_controller.servo_hand(finger_velocites, self.robot)
    
    def render(self):
        if self.visual_ob:
            return self._get_latest_cam()
        else:
            print("set visual_ob to true")
            return None

    # can't use without ros
    def stabilize_jaco(self):
        self._execute_joint_velocities(np.zeros(6))
        self._execute_finger_velocities(np.zeros(2))


class CameraListener:

    def __init__(self, input_topic_name, input_message_type, message_to_data):
        self.inputlock = Lock()
        self.input_topic_name = input_topic_name
        self.input_message_type = input_message_type
        self.message_to_data = message_to_data

        self.most_recent_message = None
        self.init_listener()

    def callback(self, data):
        with self.inputlock:
            self.most_recent_message = data

    def init_listener(self):
        # rospy.init_node('cam_listener', anonymous=True) #might have to delete later
        rospy.Subscriber(self.input_topic_name, self.input_message_type, self.callback)

    def get_most_recent_msg(self):
        while self.most_recent_message is None:
            print('Waiting for topic {} to publish.'.format(self.input_topic_name))
            rospy.sleep(0.02)
        with self.inputlock:
            data = self.message_to_data(self.most_recent_message)

        return data

def realsense_postprocessing(data):
    br = CvBridge()
    img = cv2.cvtColor(br.imgmsg_to_cv2(data), cv2.COLOR_BGR2RGB)

    #crop
    img = img[:, 80:-80,:]
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    return img

def webcam_postprocessing(data):
    br = CvBridge()
    img = br.imgmsg_to_cv2(data)

    #crop
    img = img[:,184:-184,:]
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    
    return img

if __name__=="__main__":
    rospy.init_node('cam_node', anonymous=True)
    # cam = CameraListener("/realsense/color/image_raw", Image, realsense_postprocessing)
    cam = CameraListener("camera/image_raw", Image, webcam_postprocessing)
    for i in range(10000):
        img = cam.get_most_recent_msg()
        
        cv2.imshow("t", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()