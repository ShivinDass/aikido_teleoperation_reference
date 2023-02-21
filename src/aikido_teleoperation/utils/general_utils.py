import numpy as np
import os
import cv2
import h5py
import matplotlib.pyplot as plt
import tf.transformations as transmethods

plt.rcParams['figure.figsize'] = (5, 3)


def apply_twist_to_transform(twist, transform, time=1.):
    transform[0:3, 3] += time * twist[0:3]
    angular_velocity = twist[3:]
    angular_velocity_norm = np.linalg.norm(angular_velocity)

    if angular_velocity_norm > 1e-3:
        angle = time * angular_velocity_norm
        axis = angular_velocity / angular_velocity_norm
        transform[0:3, 0:3] = np.dot(transmethods.rotation_matrix(angle, axis), transform)[0:3, 0:3]

    return transform

def get_largest_demo_number(data_dir):
    max_demo = -1
    for f in os.listdir(data_dir):
        max_demo = max(max_demo, int(f.split('.')[0].split('_')[-1]))
    
    return max_demo

def display(img_dict, prefix="", border_color=(0,0,0), display_imgs=True):
    # img = np.concatenate([img_dict['front_cam_ob'], cv2.rotate(img_dict['mount_cam_ob'], cv2.ROTATE_180)], axis=1)
    # img = cv2.resize(img, (896, 448), interpolation=cv2.INTER_AREA)
    img = img_dict['front_cam_ob'] #####
    if not display_imgs:
        img = (np.zeros_like(img, np.uint8) + border_color).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 5,5,5,5, cv2.BORDER_CONSTANT, value=border_color)
    cv2.imshow(prefix+'combined', img)
    cv2.waitKey(1)

    return img

def listdict2dictlist(LD):
    return {k: [dic[k] for dic in LD] for k in LD[0].keys()}

def dynamic_plotting(name, plot, ylim, threshold, plot_length=0):
    if not type(name)==list:
        name = [name]
        plot = [plot]
        ylim = [ylim]
        threshold = [threshold]

    plt.figure("Dynamic Plotting")
    plt.clf()
    for i, (n, p, y, t) in enumerate(zip(name, plot, ylim, threshold)):
        ax = plt.subplot(len(name),1,i+1)
        ax.title.set_text(n)
        if plot_length>0 and len(p)>plot_length:
            p = p[-plot_length:]
        
        ax.axhline(y=t, color='r', linestyle='--')
        if y:
            ax.set_ylim(y)
        ax.plot(p, color=(0,0,1))
    plt.pause(0.0005)

def save_trajectories(traj, username, mode):
    traj.observations = listdict2dictlist(traj.observations)
            
    for k in traj.observations.keys():
        print(k, np.array(traj.observations[k]).shape)

    data_dir = "/root/code/data/user_data/{}/{}".format(username, mode)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    demo_id = get_largest_demo_number(data_dir) + 1
    filename = os.path.join(data_dir, "demo_{}.hdf5".format(demo_id))
    print("==> Saving demonstration to file {}".format(filename))
    with h5py.File(filename, 'w') as f:
        for k in traj.keys():
            if k=='observations':
                for kk in traj[k].keys():
                    f.create_dataset(kk, data=np.array(traj[k][kk]))
            else:
                f.create_dataset(k, data=np.array(traj[k]).astype(np.float64))

def visualize_np(data, demo_id, path=None):
    dest_path = path if path else "/root/code/videos/"
    os.makedirs(dest_path, exist_ok=True)
    dest_path += "demo_{}.avi".format(demo_id)
    fps = 20
    resolution = data[0].shape[0:2]
    print(resolution)
    # write images of dataset to a video and save
    writer = cv2.VideoWriter(
        dest_path,
        cv2.VideoWriter_fourcc('M','J','P','G'),
        fps,
        resolution
    )
    for im in data:
        writer.write(im)
    writer.release()
