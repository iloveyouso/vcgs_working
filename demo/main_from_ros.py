from __future__ import print_function

# import ros2_topic_sub_bj
import numpy as np
import argparse
import grasp_estimator
import sys
import os
import glob
import mayavi.mlab as mlab
from utils.visualization_utils import *
import mayavi.mlab as mlab
from utils import utils
from data import DataLoader
import math
import numpy as np

import open3d as o3d # bjkim
import copy #BJKIM
import json #BJKIM

import socket_communication #BJKIM
import socket
import io
import struct


def visualize_with_view(pcloud, grasp_geometries, view_trajectory_json):
    """
    Visualize point cloud and grasp geometries with a specified ViewTrajectory.

    Parameters:
    - pcloud: Open3D point cloud object.
    - grasp_geometries: List of Open3D geometry objects (e.g., lines, meshes).
    - view_trajectory_json: JSON string or dictionary defining the ViewTrajectory.
    """

    view_trajectory = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.43350520948297344, 0.15658129981360958, -0.048740406956889859 ],
			"boundingbox_min" : [ 0.21683175798943533, -0.063434211103100324, -0.19384672798858676 ],
			"field_of_view" : 60.0,
			"front" : [ -0.84264386261372815, -0.38498860805038226, -0.37647721374718773 ],
			"lookat" : [ 0.33916501533465426, 0.031795626342507356, -0.11853291395750752 ],
			"up" : [ 0.28108869202880638, 0.28184702251942451, -0.91736110834854456 ],
			"zoom" : 0.6399999999999999
		}
	],
	"version_major" : 1,
	"version_minor" : 0
    }

    view = view_trajectory["trajectory"][0]

    # Initialize the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add geometries to the visualizer
    vis.add_geometry(pcloud)
    for geom in grasp_geometries:
        vis.add_geometry(geom)

    # Get the view control
    ctr = vis.get_view_control()

    # Create a CameraParameters object
    cam_params = o3d.camera.PinholeCameraParameters()

    # Extract camera vectors
    front = np.array(view["front"])
    lookat = np.array(view["lookat"])
    up = np.array(view["up"])

    # Compute the camera coordinate system
    z = front / np.linalg.norm(front)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    # Create rotation matrix
    R = np.vstack([x, y, z]).T

    # Compute distance based on bounding box and zoom
    bbox_max = np.array(view["boundingbox_max"])
    bbox_min = np.array(view["boundingbox_min"])
    bbox_size = np.linalg.norm(bbox_max - bbox_min)
    zoom = view.get("zoom", 1.0)
    distance = bbox_size / zoom

    # Compute camera position
    camera_pos = lookat + front / np.linalg.norm(front) * distance

    # Construct extrinsic matrix [R|T]
    extrinsic = np.identity(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = -R @ camera_pos


    # Apply the camera parameters
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    # Update the visualizer to apply the changes
    vis.poll_events()
    vis.update_renderer()

    # Run the visualizer
    vis.run()
    vis.destroy_window()

# #BJKIM
def create_gripper_from_panda_npy():
    """
    Loads the 'panda.npy' control points and applies the same modifications
    shown in your snippet to form a simple open3d LineSet (like a skeleton).
    """
    # Load and modify the control points just as in your snippet:
    grasp_pc = np.squeeze(utils.get_control_point_tensor(1, False), 0)
    grasp_pc[2, 2] = 0.059
    grasp_pc[3, 2] = 0.059

    mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])

    modified_grasp_pc = []
    modified_grasp_pc.append(np.zeros((3, ), np.float32))
    modified_grasp_pc.append(mid_point)
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[4])
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[3])
    modified_grasp_pc.append(grasp_pc[5])
    grasp_pc = np.asarray(modified_grasp_pc)

    # Create a LineSet from these 7 points
    # We'll just connect them in sequence to visualize the "C" shape
    lines = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6]]
    line_colors = [[1.0, 0.0, 0.0] for _ in range(len(lines))]  # all red

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(grasp_pc)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    return line_set

def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='checkpoints/gan_pretrained/')
    parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default='checkpoints/evaluator_pretrained/')
    parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    parser.add_argument('--refine_steps', type=int, default=25)

    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.80,
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )

    parser.add_argument('--target_pc_size', type=int, default=1024)
    parser.add_argument('--num_grasp_samples', type=int, default=200)
    parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=60,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    parser.add_argument('--train_data', action='store_true')
    opts, _ = parser.parse_known_args()
    if opts.train_data:
        parser.add_argument('--dataset_root_folder',
                            required=True,
                            type=str,
                            help='path to root directory of the dataset.')
    return parser


def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y, x, :]

    return pc_colors


def backproject(depth_cv,
                intrinsic_matrix,
                return_finite_depth=True,
                return_selection=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X

# Sample down the point cloud using FPS method
def compute_centroid(point_cloud):
    """
    Computes the centroid of a point cloud.

    Parameters:
    - point_cloud (np.ndarray): Input point cloud of shape (N, 3).

    Returns:
    - centroid (np.ndarray): The centroid coordinates of shape (3,).
    """
    mask = ~np.isnan(point_cloud).any(axis=1)
    clean_pc = point_cloud[mask]
    centroid = np.mean(clean_pc, axis=0)
    return centroid

def filter_point_cloud_within_cube(point_cloud, centroid, cube_size=0.3):
    """
    Filters the point cloud to include only points within a centered cube.

    Parameters:
    - point_cloud (np.ndarray): Input point cloud of shape (N, 3).
    - centroid (np.ndarray): The centroid coordinates of shape (3,).
    - cube_size (float): The length of the cube's edge. Default is 3.0 meters.

    Returns:
    - filtered_pc (np.ndarray): The filtered point cloud within the cube.
    """
    half_size = cube_size / 2.0
    min_bound = centroid - half_size
    max_bound = centroid + half_size

    # Create a boolean mask for points within the cube on all axes
    mask = np.all((point_cloud >= min_bound) & (point_cloud <= max_bound), axis=1)
    filtered_pc = point_cloud[mask]
    return filtered_pc

def receive_pointcloud_from_socket(host='0.0.0.0', port=65433):
    """
    소켓 서버를 열어 pointcloud_publisher.py에서 전송한 npy 형식의 포인트 클라우드 데이터를 수신합니다.
    4바이트 헤더로 데이터 길이를 받고, 그 길이만큼의 데이터를 np.load로 읽어 numpy 배열로 복원합니다.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print(f"Listening for pointcloud data on {host}:{port} ...")
    conn, addr = s.accept()
    print(f"Connected by {addr}")

    # 4바이트 헤더 수신(데이터 길이)
    header = conn.recv(4)
    if len(header) < 4:
        print("Failed to receive message header")
        conn.close()
        s.close()
        return None
    data_length = struct.unpack('I', header)[0]

    data = b''
    while len(data) < data_length:
        packet = conn.recv(data_length - len(data))
        if not packet:
            break
        data += packet

    conn.close()
    s.close()

    # np.load로 numpy 배열 복원
    buffer = io.BytesIO(data)
    try:
        npy_array = np.load(buffer)
        print(f"Received pointcloud numpy array of shape {npy_array.shape}")
    except Exception as e:
        print(f"Error loading numpy array: {e}")
        npy_array = None

    return npy_array

def main(args):
    parser = make_parser()
    args = parser.parse_args()
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(
        args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                               grasp_evaluator_args, args)
    
    print("Using socket to receive pointcloud...")
    object_pc = receive_pointcloud_from_socket(host='0.0.0.0', port=65433)
    if object_pc is None:
        print("No pointcloud received from socket. Exiting.")
        return
    pc = object_pc.copy()
    pc_colors = object_pc.copy()
    centroid = compute_centroid(object_pc)
    filtered_pc = filter_point_cloud_within_cube(object_pc, centroid, cube_size=0.2)
    if len(filtered_pc) == 0:
        print("No points found within the cube. Exiting.")
        return
    object_pc = filtered_pc
    pc = filtered_pc
    pc_colors = filtered_pc
    generated_grasps, generated_scores = estimator.generate_and_refine_grasps(object_pc)
    print(f"Generated {len(generated_grasps)} grasps from received pointcloud.")

    # grasp filtering: z 위치 및 gripper z-axis 각도 조건 적용
    z_threshold = -0.125
    filtered_grasps = []
    filtered_scores = []
    for grasp, score in zip(generated_grasps, generated_scores):
        if grasp[2, 3] >= z_threshold:
            filtered_grasps.append(grasp)
            filtered_scores.append(score)
    desired_z = np.array([0, 0, -1], dtype=float)
    angle_limit = math.radians(20)
    temp_grasps, temp_scores = [], []
    for g, s in zip(filtered_grasps, filtered_scores):
        z_axis = g[:3, 2] / np.linalg.norm(g[:3, 2])
        angle = math.acos(np.clip(np.dot(z_axis, desired_z), -1.0, 1.0))
        if angle <= angle_limit:
            temp_grasps.append(g)
            temp_scores.append(s)
    generated_grasps = temp_grasps
    generated_scores = temp_scores

    if generated_grasps and generated_scores:
        generated_grasps = [generated_grasps[0]]
        generated_scores = [generated_scores[0]]
    print("Selected grasp(s):", generated_grasps)

    socket_communication.send_grasp_data(host='localhost', port=65432,
                                         grasps=generated_grasps, scores=generated_scores)

    # Open3D를 이용한 시각화
    pcloud = o3d.geometry.PointCloud()
    pcloud.points = o3d.utility.Vector3dVector(pc)
    pc_colors_float = pc_colors.astype(np.float64) / 255.0
    pcloud.colors = o3d.utility.Vector3dVector(pc_colors_float)
    two_finger_gripper = create_gripper_from_panda_npy()
    grasp_geometries = []
    for grasp_matrix, score in zip(generated_grasps, generated_scores):
        gripper_geom = copy.deepcopy(two_finger_gripper)
        gripper_geom.paint_uniform_color([1.0, 0.0, 0.0])
        gripper_geom.transform(grasp_matrix)
        grasp_geometries.append(gripper_geom)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcloud] + grasp_geometries + [coordinate_frame])




if __name__ == '__main__':
    main(sys.argv[1:])
