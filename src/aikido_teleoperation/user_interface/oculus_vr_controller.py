import numpy as np
from oculus_reader.reader import OculusReader


class OculusVRController:
    def __init__(self, vr_speed = 1, finger_speed = 0.3):
        self.vr = OculusReader()

        self.vr_speed = vr_speed
        self.finger_speed = finger_speed

        self.prev_handle_press = False
        self.prev_gripper_press = False
        self.prev_vr_pos = None
        self.prev_sim_pos = None

    def get_action(self):
        """
        Gets the corresponding action from Oculus Quest2
        """
        
        default_action_pos = np.array([0, 0, 0])
        
        (
            vr_poses,
            handle_press,
            gripper_press,
            a_press,
            b_press,
        ) = self.get_pose_and_button()

        action_pos = default_action_pos.copy()
        
        if handle_press:
            if self.prev_handle_press:
                # compute action
                vr_pos = vr_poses[:3, 3]
                action_pos = self.get_cartesian_action(vr_pos)
            else:
                # reset reference vr pose
                self.prev_vr_pos = vr_poses[:3, 3]

        self.prev_handle_press = handle_press
        self.prev_gripper_press = gripper_press
        
        action = np.zeros(5)
        action[:3] = action_pos
        if a_press:
            action[3:5] = self.finger_speed
        elif b_press:
            action[3:5] = -self.finger_speed
        else:
            action[3:5] = 0
        return action, gripper_press, handle_press

    def get_pose_and_button(self):
        poses, buttons = self.vr.get_transformations_and_buttons()

        handle_press = False
        vr_poses = None
        gripper_press = False

        handle_press = buttons.get("RTr", False)
        gripper_press = buttons.get("RG", False)
        a_press = buttons.get("A", False) or buttons.get("X", False)
        b_press = buttons.get("B", False) or buttons.get("Y", False)

        vr_poses = poses.get("r", None)

        return vr_poses, handle_press, gripper_press, a_press, b_press

    def get_cartesian_action(self, vr_pos):
        rel_vr_pos = self.prev_vr_pos - vr_pos

        # relative movement speed between VR and simulation
        rel_vr_pos *= self.vr_speed

        # swap y, z axes
        rel_vr_pos[0], rel_vr_pos[1], rel_vr_pos[2] = rel_vr_pos[2], rel_vr_pos[0], -rel_vr_pos[1]
        
        return rel_vr_pos

    def stop(self):
        print("Stopping Oculus.")
        self.vr.stop()