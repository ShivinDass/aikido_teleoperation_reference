import argparse
import os
import h5py
import cv2
import time
import numpy as np
from utils.general_utils import get_largest_demo_number, listdict2dictlist, display, visualize_np
from jaco_gym import JacoGym
from user_interface.oculus_vr_controller import OculusVRController
from user_interface.ds4_controller import DS4Controller

import rospy
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
from jaco_scene import JacoScene

in_scene = [
        "table",
        "sink",
        "dish_rack",
        "oven",
        
        # bowls
        # "gray_bowl",
        "black_bowl",
        "blue_bowl",

        # plates
        # "gray_plate",
        # "white_plate",
        
        # cups
        # "green_cup",
        # "yellow_cup",
        
        # bread
        # "long_bread",
        # "square_bread",
        
        # dairy
        "butter_dairy",
        "milk_dairy",
        
        # meats
        # "burger_meat",
        # "steak_meat",
        
        # fruit
        "apple_fruit",
        # "orange_fruit",
    ]

def collect_data(args):
    record_data = True
    env = JacoGym(visual_ob=True, frequency=10)
    controller = DS4Controller()
    scene = JacoScene(in_scene=in_scene, switch_out_freq=15)
    
    def shutdown_helper():
        if type(controller) == OculusVRController:
            controller.stop()
    
    while not rospy.is_shutdown():
        obs = env.reset()

        data = {
            'actions': [],
            'observations': [],
            'terminals': [],
            'prompts': [],
            'scene_metadata': []
        }
        
        time_step = 0
        current_prompt = str(scene.prompt())
        print("Set Scene with,\n"+str(scene.in_scene))
        print("Current Task:", current_prompt)

        prev_skip = False
        prev_next = False
        while not rospy.is_shutdown():
            controls = controller.get_action()

            obs, rew, _, info = env.step(controls.action)
            # print("EEF pos:", obs["ee_cartesian_pos_ob"][:3])
            if record_data:
                data['actions'].append(controls.action)
                data['terminals'].append(controls.done)
                data['observations'].append(obs)
                data['scene_metadata'].append(scene.get_info())

            time_step += 1
            if controls.next_prompt and not prev_next:
                data['prompts'] = data['prompts'] + ([current_prompt]*(len(data['actions']) - len(data['prompts'])))
                scene.update()
                current_prompt = str(scene.prompt())
                print("Current Task:", current_prompt)
                
            elif controls.skip_prompt and not prev_skip:
                data['prompts'] = data['prompts'] + (['skipped']*(len(data['actions']) - len(data['prompts'])))
                current_prompt = str(scene.prompt())
                print("Current Task:", current_prompt)

            prev_skip = controls.skip_prompt
            prev_next = controls.next_prompt
            if controls.done:
                print(scene.in_scene)
                data['prompts'] = data['prompts'] + (['skipped']*(len(data['actions']) - len(data['prompts'])))
                break
            
        if record_data and not rospy.is_shutdown():
            data['observations'] = listdict2dictlist(data['observations'])
            data['scene_metadata'] = listdict2dictlist(data['scene_metadata'])
            
            for k in data:
                if k == 'observations':
                    for o in data[k]:
                        print("{} : {}".format(o, np.array(data[k][o]).shape))
                else:
                    print("{} : {}".format(k, np.array(data[k]).shape))

            data_dir = "/root/code/data/pick_apple"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            demo_id = get_largest_demo_number(data_dir) + 1
            filename = os.path.join(data_dir, "demo_{}.h5".format(demo_id))
            print("==> Saving demonstration to file {}".format(filename))
            with h5py.File(filename, 'w') as f:
                for k in data['observations']:
                    f.create_dataset(k, data=np.array(data['observations'][k]))
                f.create_dataset('actions', data=np.array(data['actions']))
                f.create_dataset('terminals', data=np.array(data['terminals']))
                f.create_dataset('prompts', data=np.array(data['prompts'], dtype='S'))

                for k in data['scene_metadata']:
                    f.create_dataset('info/{}'.format(k), data=np.array(data['scene_metadata'][k], dtype='S'))
            
            visualize_np(np.array(data['observations']['front_cam_ob']), demo_id)
    rospy.on_shutdown(shutdown_helper)



if __name__ == "__main__":
    
    rospy.init_node('aikido_collect_data', anonymous=True)
    roscpp_init('aikido_collect_data', [])

    collect_data(None)