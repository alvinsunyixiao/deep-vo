import numpy as np
from skimage import io as skio
from skimage import transform as sktran

from utils.pose3d import Pose3D, Rot3D

def load_data(path):
    img_file = path + "_img.png"
    depth_file = path + "_depth.npy"
    pose_file = path + "_pose.npy"

    img = skio.imread(img_file)
    img = img[..., :-1] / 255.
    depth = np.load(depth_file)
    depth = depth.reshape(img.shape[:-1]).astype(np.float32)
    pose = np.load(pose_file)

    img = sktran.resize(img, (160, 256))
    depth = sktran.resize(depth, (160, 256))

    ned_T_edn = Pose3D.identity()
    ned_T_edn.R = Rot3D.from_matrix([
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.],
    ])

    return {
        "img": img.astype(np.float32),
        "depth": depth,
        "pose": Pose3D.from_storage(pose) @ ned_T_edn,
    }


