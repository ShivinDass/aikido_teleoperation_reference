from __future__ import division

import threading
import time

import numpy as np

import actionlib
import pr_control_msgs.msg
from trajectory_msgs.msg import JointTrajectoryPoint


class ServoController(object):
    def __init__(self, ifsim, period, watchdog_timeout, robot):
        self.sim = ifsim
        self.robot = robot
        self.hand = robot.get_hand()
        self.period = period
        self.num_dofs = robot.get_num_arm_dofs()
        self.finger_dofs = self.hand.get_num_finger_dofs()
        # import pdb
        # pdb.set_trace()
        self.q_dot = np.zeros(self.num_dofs)
        self.q_dot_f =np.zeros(self.finger_dofs)

        self.running = False
        self.watchdog = time.time()
        self.watchdog_timeout = watchdog_timeout

        self.timer = None
        self.mutex = threading.Lock()
        self.event = threading.Event()

        self.joints = ['j2n6s200_joint_1', 'j2n6s200_joint_2', 'j2n6s200_joint_3', 'j2n6s200_joint_4', 'j2n6s200_joint_5', 'j2n6s200_joint_6']
        self.fingers = ['j2n6s200_joint_finger_1', 'j2n6s200_joint_finger_2']

        self.start()

    def set_period(self, period):
        self.period = period

    def servo_arm(self, joint_vels, robot):
        with self.mutex:
            if (joint_vels != np.zeros(self.num_dofs)).any():
                self.q_dot = np.array(joint_vels, dtype='float')
                self.watchdog = time.time()
                self.running = True
            else:
                self.q_dot = np.array(joint_vels, dtype='float')
                if self.sim:
                    self.running = False

    def servo_hand(self, finger_vels, robot):
        with self.mutex:
            if (finger_vels != np.zeros(self.finger_dofs)).any():
                self.q_dot_f = np.array(finger_vels, dtype='float')
                self.watchdog = time.time()
                self.running = True
            else:
                self.q_dot_f = np.array(finger_vels, dtype='float')
                if self.sim:
                    self.running = False

    def start(self):
        self.thread = threading.Thread(target=self.step)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join()

    def step(self):
        while True:
            start_time = time.time()
            with self.mutex:
                # Copy the velocity for thread safety
                q_dot = self.q_dot.copy()
                q_dot_f = self.q_dot_f.copy()
                running = self.running

                # stop servoing when the watchdog times out.
                if running and start_time - self.watchdog > self.watchdog_timeout:
                    self.q_dot = np.zeros(self.num_dofs)
                    self.running = False
            if running:
                q = self.robot.get_arm_positions()
                q += self.period * q_dot
                q_min = self.robot.get_arm_lower_limits()
                q_max = self.robot.get_arm_upper_limits()
                if ((q_min <= q).all() and (q <= q_max).all()):
                    if self.sim:
                        self.robot.set_arm_positions(q)
                    else:
                        # initialize arm client
                        arm_client = actionlib.SimpleActionClient('/velocity_controller/joint_group_command',
                                                              pr_control_msgs.msg.JointGroupCommandAction)
                        arm_client.wait_for_server()

                        # send arm velocity controls from client
                        arm_command = JointTrajectoryPoint()
                        arm_command.velocities = self.period * q_dot * 6
                        goal = pr_control_msgs.msg.JointGroupCommandGoal(joint_names=self.joints, command=arm_command)
                        arm_client.send_goal(goal)

                        # initialize finger client
                        finger_client = actionlib.SimpleActionClient('/hand_velocity_controller/joint_group_command',
                                                              pr_control_msgs.msg.JointGroupCommandAction)
                        finger_client.wait_for_server()

                        # send finger velocity controls from client
                        finger_command = JointTrajectoryPoint()
                        finger_command.velocities = self.period * q_dot_f * 15
                        goal_f = pr_control_msgs.msg.JointGroupCommandGoal(joint_names=self.fingers, command=finger_command)
                        finger_client.send_goal(goal_f)

                        # positions = self.robot.get_arm_positions()
                        # positions2 = positions.copy()
                        # positions2[0] = q[0]
                        # positions2[1] = q[1]
                        # positions2[2] = q[2]
                        # positions2[3] = q[3]
                        # positions2[4] = q[4]
                        # positions2[5] = q[5]

                        # traj = self.robot.compute_joint_space_path([(0.0, positions), (1.0, positions2)])
                        # import pdb
                        # # pdb.set_trace()
                        # self.robot.execute_trajectory(traj)
                else:
                    print("Exceeds the limits")
                    # clears all velocities
                    arm_client = actionlib.SimpleActionClient('/velocity_controller/joint_group_command',
                                                              pr_control_msgs.msg.JointGroupCommandAction)
                    arm_client.wait_for_server()
                    arm_command = JointTrajectoryPoint()
                    arm_command.velocities = q_dot * 0
                    goal = pr_control_msgs.msg.JointGroupCommandGoal(joint_names=self.joints, command=arm_command)
                    arm_client.send_goal(goal)

                    finger_client = actionlib.SimpleActionClient('/hand_velocity_controller/joint_group_command',
                                                                 pr_control_msgs.msg.JointGroupCommandAction)
                    finger_client.wait_for_server()
                    finger_command = JointTrajectoryPoint()
                    finger_command.velocities = q_dot_f * 0
                    goal_f = pr_control_msgs.msg.JointGroupCommandGoal(joint_names=self.fingers, command=finger_command)
                    finger_client.send_goal(goal_f)

                    self.running = False

            if self.event.wait(max(self.period - (time.time() - start_time), 0.)):
                break
