import copy
import os
import pickle
import h5py
import cv2
import numpy as np
from utils.general_utils import get_largest_demo_number, listdict2dictlist, display
from aikido_gym import AikidoGym

import rospy
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init

def check_lighting(img1, img2):
    print("Avg pixel img1:", np.mean(img1))
    print("Avg pixel img2:", np.mean(img2))

    print("Stddev pixel img1:", np.std(img1))
    print("Stddev pixel img2:", np.std(img2))


def rollout_traj():
    visual_ob = True
    env = JacoGym(visual_ob=visual_ob, frequency=10)
    env.reset()

    data_file = "/root/code/data/user_data/ayush/P/demo_0.hdf5"
    f = h5py.File(data_file, 'r')
    seq_end_idx = np.where(f['terminals'])[0]
    
    grip_dict = [[-0.3,-0.3], [0,0], [0.3,0.3]]
    rollout_id = 0
    for i, a in enumerate(f['actions']):
        if 'gripper' in data_file:
            print(a[-1])
            action = np.concatenate((a[:3], grip_dict[int(a[-1])]), axis=0)
        else:
            action = a
        obs, rew, _, info = env.step(action)
        
        if visual_ob:
            rend = env.render()
            cv2.imshow('t', rend['front_cam_ob'])
            cv2.waitKey(1)
            # display({'front': rend['front_cam_ob'], 'mount': rend['mount_cam_ob']})

        if f['terminals'][i]:
            break
    env.stabilize_jaco()
    rospy.sleep(2.0)


def generate_blended_demo():
    data_dir = "/root/code/data/simple/"

    blended_rollouts = [255*np.ones(shape=(224,224,3), dtype=np.uint8)]
    for n, filename in enumerate(os.listdir(data_dir)[:5]):
        data_file = os.path.join(data_dir, filename)
        
        last_frame = copy.deepcopy(blended_rollouts[-1])

        with h5py.File(data_file, 'r') as f:
            for i, img in enumerate(f['front_cam_ob']):
                if len(blended_rollouts)-1<i:
                    blended_rollouts.append(
                        cv2.addWeighted(last_frame, n/(n+1), img, 1/(n+1), gamma=0)
                    )
                else:
                    blended_rollouts[i] = cv2.addWeighted(blended_rollouts[i], n/(n+1), img, 1/(n+1), gamma=0)
        
        for j in range(i, len(blended_rollouts)):
            blended_rollouts[j] = cv2.addWeighted(blended_rollouts[j], n/(n+1), img, 1/(n+1), gamma=0)

        for img in blended_rollouts:
            cv2.imshow("B", img)
            cv2.waitKey(5)
    
    cv2.destroyAllWindows()

    vid = cv2.VideoWriter(os.path.join(data_dir, 'blended.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 30, (224,224))
    for img in blended_rollouts:
        vid.write(img)
    vid.release()
    


if __name__=='__main__':
    # generate_blended_demo()

    rospy.init_node('rollout', anonymous=True)
    roscpp_init('rollout', [])
    rollout_traj()
