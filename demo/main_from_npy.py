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


# #BJKIM
# panda_finger_model = o3d.io.read_triangle_mesh("./gripper_models/panda_gripper/finger.stl")
# panda_finger_model.compute_vertex_normals()

# #BJKIM
# def create_two_finger_c_shape_gripper():
#     """
#     Creates a slim, two-finger 'C' shaped gripper mesh using very thin boxes.
#     Returns an Open3D TriangleMesh.
#     """
#     # Base of the gripper (horizontal bar), made thinner
#     base = o3d.geometry.TriangleMesh.create_box(width=0.06, height=0.002, depth=0.002)
#     base.paint_uniform_color([0.0, 0.0, 1.0])  # Blue base
#     base.translate((0.0, 0.0, 0.0))

#     # Finger 1 (vertical bar on the left), made thinner
#     finger1 = o3d.geometry.TriangleMesh.create_box(width=0.002, height=0.04, depth=0.002)
#     finger1.paint_uniform_color([1.0, 0.0, 0.0])  # Red finger
#     finger1.translate((0.0, 0.002, 0.0))

#     # Finger 2 (vertical bar on the right), made thinner
#     finger2 = o3d.geometry.TriangleMesh.create_box(width=0.002, height=0.04, depth=0.002)
#     finger2.paint_uniform_color([1.0, 0.0, 0.0])  # Red finger
#     finger2.translate((0.058, 0.002, 0.0))

#     # Combine meshes
#     gripper = base + finger1 + finger2
#     gripper.compute_vertex_normals()
#     return gripper


def visualize_with_view(pcloud, grasp_geometries, view_trajectory_json):
    """
    Visualize point cloud and grasp geometries with a specified ViewTrajectory.

    Parameters:
    - pcloud: Open3D point cloud object.
    - grasp_geometries: List of Open3D geometry objects (e.g., lines, meshes).
    - view_trajectory_json: JSON string or dictionary defining the ViewTrajectory.
    """
    #  # Parse the JSON input
    # if isinstance(view_trajectory_json, str):
    #     try:
    #         view_trajectory = json.loads(view_trajectory_json)
    #     except Exception as e:
    #         print(e)
    #         print("JSON input was:")
    #         print(view_trajectory_json)
    #         raise
    # elif isinstance(view_trajectory_json, dict):
    #     view_trajectory = view_trajectory_json
    # else:
    #     raise TypeError("view_trajectory_json must be a JSON string or a dictionary.")

    # # Validate the presence of 'trajectory'
    # if "trajectory" not in view_trajectory:
    #     raise KeyError("The JSON does not contain a 'trajectory' key.")

    # if len(view_trajectory["trajectory"]) == 0:
    #     raise ValueError("The 'trajectory' list is empty.")

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

    # cam_params.extrinsic = extrinsic

    # # Optionally, set intrinsic parameters
    # # You can modify this part if your ViewTrajectory includes intrinsic parameters
    # intrinsic = ctr.get_intrinsic()
    # cam_params.intrinsic = intrinsic

    # Apply the camera parameters
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    # # Set the field of view
    # if "field_of_view" in view:
    #     ctr.set_field_of_view(view["field_of_view"])

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
    saved_npy_mode = True #BJKIM

    if args.train_data:
        print('traindata!s')
        # grasp_sampler_args.dataset_root_folder = args.dataset_root_folder
        # grasp_sampler_args.num_grasps_per_object = 1
        # grasp_sampler_args.num_objects_per_batch = 1
        # dataset = DataLoader(grasp_sampler_args)
        # for i, data in enumerate(dataset):
        #     generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
        #         data["pc"].squeeze())
        #     mlab.figure(bgcolor=(1, 1, 1))
        #     draw_scene(data["pc"][0],
        #                grasps=generated_grasps,
        #                grasp_scores=generated_scores)
        #     print('close the window to continue to next object . . .')
        #     mlab.show()
    elif saved_npy_mode:
        for npy_file in glob.glob(os.path.join(args.npy_folder, '*.npy')):
            # Depending on your numpy version you may need to change allow_pickle
            # from True to False.

            # import pdb; pdb.set_trace()
            data = np.load(npy_file, allow_pickle=True,
                           encoding="latin1").item()
            # depth = data['depth']
            # image = data['image']
            # K = data['intrinsics_matrix']
            # # Removing points that are farther than 1 meter or missing depth
            # # values.
            # #depth[depth == 0 or depth > 1] = np.nan

            # np.nan_to_num(depth, copy=False)
            # mask = np.where(np.logical_or(depth == 0, depth > 1))
            # depth[mask] = np.nan
            # pc, selection = backproject(depth,
            #                             K,
            #                             return_finite_depth=True,
            #                             return_selection=True)
            # pc_colors = image.copy()
            # pc_colors = np.reshape(pc_colors, [-1, 3])
            # pc_colors = pc_colors[selection, :]

            # Smoothed pc comes from averaging the depth for 10 frames and removing
            # the pixels with jittery depth between those 10 frames.
            try:
                object_pc = data['smoothed_object_pc']['xyz'].astype(np.float64)
            except:
                object_pc = data['smoothed_object_pc'].astype(np.float64)

            
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


            def farthest_point_sampling(point_cloud, num_samples):
                """
                Perform Farthest Point Sampling (FPS) on a point cloud.

                Parameters:
                - point_cloud (np.ndarray): Input point cloud of shape (N, D), where
                                            N is the number of points and D is the dimensionality (e.g., 3 for 3D).
                - num_samples (int): Number of points to sample from the point cloud.

                Returns:
                - sampled_indices (list): List of indices of the sampled points.
                """
                N, D = point_cloud.shape
                if num_samples > N:
                    raise ValueError("num_samples should be less than or equal to the number of points in the point cloud.")

                # Initialize an array to hold the indices of the sampled points
                sampled_indices = []

                # Initialize a distance array with infinity
                distances = np.full(N, np.inf)

                # Randomly select the first point and add its index to sampled_indices
                first_index = np.random.randint(0, N)
                sampled_indices.append(first_index)

                # Update the distance array with distances from the first selected point
                first_point = point_cloud[first_index]
                diff = point_cloud - first_point
                distances = np.linalg.norm(diff, axis=1)

                for _ in range(1, num_samples):
                    # Select the point with the maximum distance from the sampled points
                    next_index = np.argmax(distances)
                    sampled_indices.append(next_index)

                    # Update the distances array with the minimum distance to any of the sampled points
                    new_point = point_cloud[next_index]
                    diff = point_cloud - new_point
                    new_distances = np.linalg.norm(diff, axis=1)
                    distances = np.minimum(distances, new_distances)

                return sampled_indices
            
            # Compute the centroid of the point cloud
            centroid = compute_centroid(object_pc)

            # Filter the point cloud to include only points within a 3x3x3 cube
            filtered_pc = filter_point_cloud_within_cube(object_pc, centroid, cube_size=0.2)

            if len(filtered_pc) == 0:
                print(f"No points found within the 3x3x3 cube for file {npy_file}. Skipping.")
                continue

            num_points_to_sample = len(filtered_pc)  # Ensure not exceeding available points            
            # sampled_indices = farthest_point_sampling(filtered_pc, num_points_to_sample)

            # object_pc = object_pc[sampled_indices]
            # pc = object_pc[sampled_indices]
            # pc_colors = object_pc[sampled_indices]
            object_pc = filtered_pc
            pc = filtered_pc
            pc_colors = filtered_pc

            generated_grasps, generated_scores = estimator.generate_and_refine_grasps(object_pc)
            print(f"Generated {len(generated_grasps)} grasps and scores for file {npy_file}.")
            # print('data type for each element in generated_grasps:', type(generated_grasps[0]))

            # Filter out grasps with z position under threshold
            z_threshold = -0.125  # Set your threshold value here
            filtered_grasps = []
            filtered_scores = []

            for grasp, score in zip(generated_grasps, generated_scores):
                if grasp[2, 3] >= z_threshold:
                    filtered_grasps.append(grasp)
                    filtered_scores.append(score)
            
            # z direction of the gripper should be pointing downwards with a small angle tolerance
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

            # Select only one grasp and one score
            if generated_grasps and generated_scores:
                generated_grasps = [generated_grasps[0]]
                generated_scores = [generated_scores[0]]

            print(generated_grasps)

            socket_communication.send_grasp_data(host='localhost', port=65432, grasps=generated_grasps, scores=generated_scores)
          
            # BJKIMkim below
            # -- Open3D-based snippet --
            # Create a point cloud:
            # -- Open3D-based snippet --
            pcloud = o3d.geometry.PointCloud()
            pcloud.points = o3d.utility.Vector3dVector(pc)

            # Convert colors to float in [0, 1]
            pc_colors_float = pc_colors.astype(np.float64) / 255.0
            pcloud.colors = o3d.utility.Vector3dVector(pc_colors_float)

            two_finger_gripper = create_gripper_from_panda_npy()

            # Build a small geometry for each grasp so we can visualize the "gripper" pose.
            # If 'generated_grasps' is a list of 4x4 pose matrices, 
            # each transform can be applied to place the box properly.
            grasp_geometries = []
            for grasp_matrix, score in zip(generated_grasps, generated_scores):
                # Copy the original STL mesh so we can transform it
                gripper_geom = copy.deepcopy(two_finger_gripper)

                # Optionally, you can paint the model a uniform color
                gripper_geom.paint_uniform_color([1.0, 0.0, 0.0])  # e.g. red

                # Transform the geometry according to the grasp pose
                gripper_geom.transform(grasp_matrix)

                # Collect this geometry for later visualization
                grasp_geometries.append(gripper_geom)

            # visualisation option 2: Open3D's visualizer with camera trajectory
            # import pdb; pdb.set_trace()
            # visualize_with_view(pcloud, grasp_geometries, "./view_bj.json")


            # Add coordinate frame for reference
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcloud] + grasp_geometries + [coordinate_frame])

    else:
        for npy_file in glob.glob(os.path.join(args.npy_folder, '*.npy')):
            # Depending on your numpy version you may need to change allow_pickle
            # from True to False.

            # import pdb; pdb.set_trace()
            data = np.load(npy_file, allow_pickle=True,
                           encoding="latin1").item()
            depth = data['depth']
            image = data['image']
            K = data['intrinsics_matrix']
            # Removing points that are farther than 1 meter or missing depth
            # values.
            #depth[depth == 0 or depth > 1] = np.nan

            np.nan_to_num(depth, copy=False)
            mask = np.where(np.logical_or(depth == 0, depth > 1))
            depth[mask] = np.nan
            pc, selection = backproject(depth,
                                        K,
                                        return_finite_depth=True,
                                        return_selection=True)
            pc_colors = image.copy()
            pc_colors = np.reshape(pc_colors, [-1, 3])
            pc_colors = pc_colors[selection, :]

            # Smoothed pc comes from averaging the depth for 10 frames and removing
            # the pixels with jittery depth between those 10 frames.
            object_pc = data['smoothed_object_pc']
            generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
                object_pc)
            # mlab.figure(bgcolor=(1, 1, 1))
            # draw_scene(
            #     pc,
            #     pc_color=pc_colors,
            #     grasps=generated_grasps,
            #     grasp_scores=generated_scores,
            # )
            # print('close the window to continue to next object . . .')
            # mlab.show()

            # BJKIMkim below
            # -- Open3D-based snippet --
            # Create a point cloud:
            # -- Open3D-based snippet --
            pcloud = o3d.geometry.PointCloud()
            pcloud.points = o3d.utility.Vector3dVector(pc)

            # Convert colors to float in [0, 1]
            pc_colors_float = pc_colors.astype(np.float64) / 255.0
            pcloud.colors = o3d.utility.Vector3dVector(pc_colors_float)

            two_finger_gripper = create_gripper_from_panda_npy()

            # Build a small geometry for each grasp so we can visualize the "gripper" pose.
            # If 'generated_grasps' is a list of 4x4 pose matrices, 
            # each transform can be applied to place the box properly.
            grasp_geometries = []
            for grasp_matrix, score in zip(generated_grasps, generated_scores):
                # Copy the original STL mesh so we can transform it
                gripper_geom = copy.deepcopy(two_finger_gripper)

                # Optionally, you can paint the model a uniform color
                gripper_geom.paint_uniform_color([1.0, 0.0, 0.0])  # e.g. red

                # Transform the geometry according to the grasp pose
                gripper_geom.transform(grasp_matrix)

                # Collect this geometry for later visualization
                grasp_geometries.append(gripper_geom)

            # visualisation option 1: Open3D's built-in visualizer
            o3d.visualization.draw_geometries([pcloud] + grasp_geometries)
            
            # # visualisation option 2: Open3D's visualizer with camera trajectory
            # vis = o3d.visualization.Visualizer()
            # vis.create_window()

            # # Add geometries
            # for geo in [pcloud] + grasp_geometries:
            #     vis.add_geometry(geo)

            # # Load the single-view trajectory (your JSON file) and apply it:
            # trajectory = o3d.io.read_pinhole_camera_trajectory("./view_bj.json")
            # ctr = vis.get_view_control()
            # ctr.convert_from_pinhole_camera_parameters(trajectory.parameters[0])

            # vis.run()
            # vis.destroy_window()


if __name__ == '__main__':
    main(sys.argv[1:])
