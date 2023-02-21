import pickle
import os
import cv2
import time
import torch
import numpy as np
import sys
from utils.general_utils import display, save_trajectories, visualize_np, get_largest_demo_number
from jaco_gym import JacoGym
from user_interface.ds4_controller import DS4Controller

from r3m import load_r3m
# import mvp
# from bootstrap.rollout_utils import RolloutWrapper
from bootstrap.rollout_utils_r3m import RolloutWrapper

import rospy
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
import torchvision.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def assisted():
    visual_ob = False
    finger_speed = 0.3

    env = JacoGym(visual_ob=True, frequency=10)
    # mvp = MVPHelper()
    r3m = R3MHelper()
    controller = DS4Controller(control_mode='none', finger_speed=finger_speed)
    model = RolloutWrapper(
        "./bootstrap/real_world_10r3m_val_bcmse_lrelubn_seeddata_lr1e-4/alfred_action_model_499.pth",
        "./bootstrap/norm_constants_pick_apple_seed.pkl",
        torch.device(0),
    )

    def shutdown_helper():
        if visual_ob:
            cv2.destroyAllWindows()

    while not rospy.is_shutdown():
        obs = env.reset()
        obs_list = []

        while not rospy.is_shutdown():
            controls = controller.get_action()
            
            if not controls.override_control:
                controls.action = np.zeros_like(controls.action)

            obs = r3m.preprocess_obs(obs)
            obs = model.preprocess_ob(obs)
            # r3m.preprocess_obs(obs)
            # if bc_config.normalized==True:
            #     obs[-15:] = (obs[-15:]-inp_mean)/inp_stddev

            twist = model(obs)
            print(twist)

            # if bc_config.normalized==True:
            #     twist = twist*out_stddev + out_mean

            if (np.linalg.norm(controls.action)==0) and not controls.override_control:
                controls.action[:3] = twist[:3]
                auto_action = True
                
                if twist[-1]==0:
                    controls.action[-2:] = -finger_speed
                elif twist[-1]==1:
                    controls.action[-2:] = 0
                else:
                    controls.action[-2:] = finger_speed
            else:
                model.reset_inference()

            obs, rew, _, info = env.step(controls.action)
            
            # obs_list.append(obs['front_cam_ob'])#np.concatenate((obs['front_cam_ob'], obs['mount_cam_ob']), axis=1))
            if controls.done:
                break
        # demo_id = get_largest_demo_number(path) + 1
        # if len(obs_list) > 0:
        #     visualize_np(np.array(obs_list), demo_id, path)
        
    rospy.on_shutdown(shutdown_helper)

def load_bc_model(epoch=None):
    if epoch is not None:
        bc_config.epoch = epoch
    model = model_class(bc_config)
    
    path = os.path.join(bc_config.experiment_dir, 'weights')
    model_epoch = bc_config.epoch if 'epoch' in bc_config else 'latest'
    CheckpointHandler.load_weights(CheckpointHandler.get_resume_ckpt_file(model_epoch, path), model)
    
    model.to(bc_config.device)
    return model

def get_normalizing_constants():
    norm_file_path = os.path.join(bc_config.experiment_dir, 'norm_constants.pkl')

    with open(norm_file_path, 'rb') as f:
        norm_factors = pickle.load(f)
    
    return norm_factors.observations_mean, norm_factors.observations_stddev, \
            norm_factors.actions_mean, norm_factors.actions_stddev 

class R3MHelper:

    def __init__(self) -> None:
        self.r3m = load_r3m('resnet50')
        self.r3m.eval()
        self.r3m.to(device)

    def preprocess_obs(self, obs):
        with torch.no_grad():
            front_cam_emb = self.r3m(torch.tensor(obs['front_cam_ob'].transpose(2,0,1).reshape(1,3,224,224))).data.cpu().numpy().copy().squeeze()
            mount_cam_emb = self.r3m(torch.tensor(obs['mount_cam_ob'].transpose(2,0,1).reshape(1,3,224,224))).data.cpu().numpy().copy().squeeze()

            # front_cam_emb = self.transform(front_cam_emb)
            # mount_cam_emb = self.transform(mount_cam_emb)

        observations = np.concatenate((front_cam_emb, mount_cam_emb), axis=0)
        for obs_key in ['ee_cartesian_pos_ob', 'ee_cartesian_vel_ob']:
            observations = np.concatenate((observations, obs[obs_key]), axis=0)
        
        observations = np.concatenate((observations, obs['joint_pos_ob'][-2:]), axis=0)
        return observations

class MVPHelper:
    
    def __init__(self) -> None:
        self.mvp = mvp.load('vitb-mae-egosoup').eval().to(device)
        self.mvp.freeze()
        self.transform = T.Compose(
            [
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def preprocess_obs(self, obs):
        with torch.no_grad():
            f_embed = torch.tensor(obs['front_cam_ob'].transpose(2,0,1).reshape(1,3,224,224)).float().to(device)/255
            m_embed = torch.tensor(obs['mount_cam_ob'].transpose(2,0,1).reshape(1,3,224,224)).float().to(device)/255

            f_embed = self.transform(f_embed)
            m_embed = self.transform(m_embed)

            front_cam_emb = self.mvp(f_embed).data.cpu().numpy().copy().squeeze()
            mount_cam_emb = self.mvp(m_embed).data.cpu().numpy().copy().squeeze()
        
        observations = np.concatenate((front_cam_emb, mount_cam_emb), axis=0)
        for obs_key in ['ee_cartesian_pos_ob', 'ee_cartesian_vel_ob']:
            observations = np.concatenate((observations, obs[obs_key]), axis=0)
        
        observations = np.concatenate((observations, obs['joint_pos_ob'][-2:]), axis=0)
        return observations

if __name__ == "__main__":
    rospy.init_node('aikido_assisted', anonymous=True)
    roscpp_init('aikido_assisted', [])
    assisted()