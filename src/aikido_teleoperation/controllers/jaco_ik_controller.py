"""
NOTE: requires pybullet module.

Run `pip install pybullet==1.9.5`.
"""

import numpy as np

try:
    import pybullet as p
except ImportError:
    raise Exception(
        "Please make sure pybullet is installed. Run `pip install pybullet==1.9.5`"
    )
from os.path import join as pjoin

from . import transform_utils as T
from .controller import Controller


class JacoIKController(Controller):
    """
    Inverse kinematics for the Jaco robot, using Pybullet and the urdf description
    files. Loads a jaco robot into an internal Pybullet simulation, and uses it to
    do inverse kinematics computations.
    """

    def __init__(self, bullet_data_path, robot_jpos_getter, hand_jpos_getter, min_pos_lim=None, max_pos_lim=None):
        """
        Args:
            bullet_data_path (str): base path to bullet data.

            robot_jpos_getter (function): function that returns the joint positions of
                the robot to be controlled as a numpy array.
        """

        # path to data folder of bullet repository
        self.bullet_data_path = bullet_data_path

        # returns current robot joint positions
        self.robot_jpos_getter = robot_jpos_getter
        self.hand_jpos_getter = hand_jpos_getter

        #min and max cartesian pos lims for safety boundry
        self.min_pos_lim = np.array(min_pos_lim)
        self.max_pos_lim = np.array(max_pos_lim)

        # Do any setup needed for Inverse Kinematics.
        self.setup_inverse_kinematics()

        # Should be in (0, 1], smaller values mean less sensitivity.
        self.user_sensitivity = .3
        # Set threshold for when goal is reached
        self.goal_thresh = 0.003

        self.sync_state()

    def get_control(self, dpos=None, rotation=None):
        """
        Returns joint velocities to control the robot after the target end effector
        position and orientation are updated from arguments @dpos and @rotation.
        If no arguments are provided, joint velocities will be computed based
        on the previously recorded target.

        Args:
            dpos (numpy array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (numpy array): a rotation matrix of shape (3, 3) corresponding
                to the desired orientation of the end effector.

        Returns:
            velocities (numpy array): a flat array of joint velocity commands to apply
                to try and achieve the desired input control.
        """

        # Sync joint positions for IK.
        self.sync_ik_robot(self.robot_jpos_getter())
        
        # Compute new target joint positions if arguments are provided
        if (dpos is not None) and (rotation is not None):
            self.commanded_joint_positions = self.joint_positions_for_eef_command(
                dpos, rotation
            )

        # P controller from joint positions (from IK) to velocities
        velocities = np.zeros(6)
        deltas = self._get_current_error(
            self.robot_jpos_getter(), self.commanded_joint_positions
        )    
    
        for i, delta in enumerate(deltas):
            velocities[i] = -6. * delta  # -2. * delta
        velocities = self.clip_joint_velocities(velocities)

        self.commanded_joint_velocities = velocities
        return velocities

    def sync_state(self):
        """
        Syncs the internal Pybullet robot state to the joint positions of the
        robot being controlled.
        """

        # sync IK robot state to the current robot joint positions
        self.sync_ik_robot(self.robot_jpos_getter())

        # make sure target pose is up to date
        self.ik_robot_target_pos, self.ik_robot_target_orn = (
            self.ik_robot_eef_joint_cartesian_pose()
        )

    def setup_inverse_kinematics(self):
        """
        This function is responsible for doing any setup for inverse kinematics.
        Inverse Kinematics maps end effector (EEF) poses to joint angles that
        are necessary to achieve those poses.
        """

        # Set up a connection to the PyBullet simulator.
        p.connect(p.DIRECT)
        p.resetSimulation()

        # get paths to urdfs
        self.robot_urdf = pjoin(
            self.bullet_data_path, "jaco_description/urdf/jaco_arm.urdf"
        )

        # load the urdfs
        self.ik_robot = p.loadURDF(self.robot_urdf, (0, 0, 0), useFixedBase=1)

        # Simulation will update as fast as it can in real time, instead of waiting for
        # step commands like in the non-realtime case.
        p.setRealTimeSimulation(1)

    def sync_ik_robot(self, joint_positions, simulate=False, sync_last=True):
        """
        Force the internal robot model to match the provided joint angles.

        Args:
            joint_positions (list): a list or flat numpy array of joint positions.
            simulate (bool): If True, actually use physics simulation, else
                write to physics state directly.
            sync_last (bool): If False, don't sync the last joint angle. This
                is useful for directly controlling the roll at the end effector.
        """

        num_joints = len(joint_positions)
        if not sync_last:
            num_joints -= 1
        for i in range(num_joints):
            if simulate:
                p.setJointMotorControl2(
                    self.ik_robot,
                    i,
                    p.POSITION_CONTROL,
                    targetVelocity=0,
                    targetPosition=joint_positions[i],
                    force=500,
                    positionGain=0.5,
                    velocityGain=1.,
                )
            else:
                p.resetJointState(self.ik_robot, i, joint_positions[i], 0)

    def ik_robot_eef_joint_cartesian_pose(self):
        """
        Returns the current cartesian pose of the last joint of the ik robot with respect to the base frame as
        a (pos, orn) tuple where orn is a x-y-z-w quaternion
        """
        eef_pos_in_world = np.array(p.getLinkState(self.ik_robot, 5)[0])
        eef_orn_in_world = np.array(p.getLinkState(self.ik_robot, 5)[1])
        eef_pose_in_world = T.pose2mat((eef_pos_in_world, eef_orn_in_world))

        base_pos_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[0])
        base_orn_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[1])
        base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        eef_pose_in_base = T.pose_in_A_to_pose_in_B(
            pose_A=eef_pose_in_world, pose_A_in_B=world_pose_in_base
        )

        return T.mat2pose(eef_pose_in_base)

    def inverse_kinematics(self, target_position, target_orientation, rest_poses=None):
        """
        Helper function to do inverse kinematics for a given target position and
        orientation in the PyBullet world frame.

        Args:
            target_position: A tuple, list, or numpy array of size 3 for position.
            target_orientation: A tuple, list, or numpy array of size 4 for
                a orientation quaternion.
            rest_poses: (optional) A list of size @num_joints to favor ik solutions close by.

        Returns:
            A list of size @num_joints corresponding to the joint angle solution.
        """

        if rest_poses is None:
            ik_solution = list(
                p.calculateInverseKinematics(
                    self.ik_robot,
                    5,
                    target_position,
                    targetOrientation=target_orientation,
                    restPoses=[0, 0, 0, 0, 0, 0],
                    jointDamping=[0.2] * 6,
                )
            )
        else:
            ik_solution = list(
                p.calculateInverseKinematics(
                    self.ik_robot,
                    5,
                    target_position,
                    targetOrientation=target_orientation,
                    lowerLimits=[-6.28, -6.28, -6.28, -6.28, -6.28, -6.28],
                    upperLimits=[6.28, 6.28, 6.28, 6.28, 6.28, 6.28],
                    jointRanges=[12.56, 12.56, 12.56, 12.56, 12.56, 12.56],
                    restPoses=rest_poses,
                    jointDamping=[0.2] * 6,
                )
            )
        return ik_solution

    def bullet_base_pose_to_world_pose(self, pose_in_base):
        """
        Convert a pose in the base frame to a pose in the world frame.

        Args:
            pose_in_base: a (pos, orn) tuple.

        Returns:
            pose_in world: a (pos, orn) tuple.
        """
        pose_in_base = T.pose2mat(pose_in_base)

        base_pos_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[0])
        base_orn_in_world = np.array(p.getBasePositionAndOrientation(self.ik_robot)[1])
        base_pose_in_world = T.pose2mat((base_pos_in_world, base_orn_in_world))

        pose_in_world = T.pose_in_A_to_pose_in_B(
            pose_A=pose_in_base, pose_A_in_B=base_pose_in_world
        )
        return T.mat2pose(pose_in_world)

    def joint_positions_for_eef_command(self, dpos, rotation):
        """
        This function runs inverse kinematics to back out target joint positions
        from the provided end effector command.

        Same arguments as @get_control.

        Returns:
            A list of size @num_joints corresponding to the target joint angles.
        """        

        new_pos = self.ik_robot_target_pos + (dpos * self.user_sensitivity)
        self.ik_robot_target_orn = T.mat2quat(rotation)

         # keep the target pose inside the cuboid defined by pos limits and safety constraints
        self.ik_robot_target_pos = self.apply_pos_limits(self.ik_robot_target_pos, new_pos)

        # convert from target pose in base frame to target pose in bullet world frame
        world_targets = self.bullet_base_pose_to_world_pose(
            (self.ik_robot_target_pos, self.ik_robot_target_orn)
        )
        
        world_targets = (world_targets[0], np.array(p.getLinkState(self.ik_robot, 5)[1]))
        rest_poses = [0, np.pi*3/4, -np.pi*1/4, 0, 0, 0]

        for bullet_i in range(20):
            arm_joint_pos = self.inverse_kinematics(
                world_targets[0], world_targets[1], rest_poses=None
            )
            self.sync_ik_robot(arm_joint_pos, sync_last=True)
        # print("result:", arm_joint_pos)
        return arm_joint_pos

    def _get_current_error(self, current, set_point):
        """
        Returns an array of differences between the desired joint positions and current
        joint positions. Useful for PID control.

        Args:
            current: the current joint positions.
            set_point: the joint positions that are desired as a numpy array.

        Returns:
            the current error in the joint positions.
        """
        error = current - set_point
        return error

    def clip_joint_velocities(self, velocities):
        """
        Clips joint velocities into a valid range.
        """
        for i in range(len(velocities)):
            if velocities[i] >= 1.0:
                velocities[i] = 1.0
            elif velocities[i] <= -1.0:
                velocities[i] = -1.0
        return velocities

    def reached_goal(self):
        dist_from_goal = np.linalg.norm(self._get_current_error(self.robot_jpos_getter(), self.commanded_joint_positions))
        if dist_from_goal < self.goal_thresh:
            return True
        return False

    def apply_pos_limits(self, pos, npos):
        min_pos_lim, max_pos_lim = self.get_safety_limits(pos, npos)

        for i, p in enumerate(npos):
            if min_pos_lim[i]:
                npos[i] = max(min_pos_lim[i], npos[i])
            if max_pos_lim[i]:
                npos[i] = min(max_pos_lim[i], npos[i])
        return npos

    def get_safety_limits(self, pos, npos):

        min_pos_lim = self.min_pos_lim.copy()
        max_pos_lim = self.max_pos_lim.copy()

        # setting min z-pos according to gripper open/closed
        if self.hand_jpos_getter()[0]<1.0:
            # gripper open: the table has a small gradient
            min_pos_lim[2] = max(min_pos_lim[2], \
                        0.18 - (0.18-min_pos_lim[2])*(pos[1]-min_pos_lim[1])/(max_pos_lim[1]-min_pos_lim[1]))
        else:
            # gripper closed: Set min z-pos when gripper closed 
            if pos[2]<=0.245:
                min_pos_lim[2] = max(min_pos_lim[2], pos[2])
            else:
                min_pos_lim[2] = max(min_pos_lim[2], 0.245)

        # white bowl safety (bowl height:0.215, bowl end:-0.41)
        # if pos[1]>=-0.41 and pos[2]>=0.215 and npos[2]<0.215 and pos[0]<0.18:
        #     min_pos_lim[2] = max(min_pos_lim[2], 0.215)
        # if pos[2]<0.215 and pos[1]<=-0.41 and npos[1]>-0.41 and pos[0]<0.18:
        #     max_pos_lim[1] = min(max_pos_lim[1], -0.41)
        # if pos[2]<0.215 and pos[0]>=0.18 and npos[0]<0.18 and pos[1]>=-0.41:
        #     min_pos_lim[0] = max(min_pos_lim[0], 0.18)

        # # oven tray safety (tray height:0.32, tray start:-0.12)
        # if pos[2]<0.32 and pos[0]>=-0.12 and npos[0]<-0.12 and pos[1]>-0.58: 
        #     min_pos_lim[0] = max(min_pos_lim[0], -0.12)
        # if pos[0]<-0.12 and pos[2]>=0.32 and npos[2]<0.32 and pos[1]>-0.58: 
        #     min_pos_lim[2] = max(min_pos_lim[2], 0.32)
        # if pos[2]<0.32 and pos[1]<=-0.58 and npos[1]>-0.58 and pos[0]<-0.12: 
        #     max_pos_lim[1] = min(max_pos_lim[1], -0.58)

        return min_pos_lim, max_pos_lim


    def bullet_get_current_joint_pos(self):
        cur_joint_states = p.getJointStates(self.ik_robot, [i for i in range(6)])
        cur_joint_pos = [cur_joint_states[i][0] for i in range(6)]
        # cur_joint_vel = [cur_joint_states[i][1] for i in range(6)]
        return cur_joint_pos

    def get_cartesian_info(self, qd):
        '''
            Return the eef cartesian pose and velocity in world frame.
            Code referred from: https://github.com/bulletphysics/bullet3/issues/2429
        '''
        q = self.robot_jpos_getter()
        self.sync_ik_robot(self.robot_jpos_getter())

        (ee_pos, ee_rot) = p.getLinkState(self.ik_robot, 5)[:2]

        q = self.bullet_get_current_joint_pos() #only get joint pose since velocity is 0 in sim
        (jac_t, jac_r) = p.calculateJacobian(self.ik_robot, 5, [0]*3, q, [0]*6, [0]*6)

        ee_pos_vel = np.array(np.matmul(jac_t, np.array(qd).reshape((6,1))).T[0])
        ee_rot_vel = np.array(np.matmul(jac_r, np.array(qd).reshape((6,1))).T[0])
        
        return ee_pos, ee_rot, ee_pos_vel, ee_rot_vel