import os
import sys
import numpy as np
import time
import torch
import open3d as o3d
from copy import deepcopy
# from graspnetAPI.graspnet_eval import GraspGroup
from graspnetAPI import GraspGroup, Grasp
# os.environ["OPEN3D_RENDERING_BACKEND"] = "egl"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector, FrankaCollisionDetector
from scipy.spatial.transform import Rotation as R, Slerp
import cv2
import plotly.graph_objects as go
# import debugpy
# debugpy.listen(('0.0.0.0', 5681))
# print('Waiting for debugger attach')
# debugpy.wait_for_client()
# from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask



def is_point_in_mask(self, point: np.array, mask: np.array):
    """Check if a point is in a mask."""
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    # K = np.array([455.6827087402344, 0.0, 326.03302001953125, 0.0, 454.86822509765625, 180.9109649658203, 0.0, 0.0, 1.0]).reshape((3, 3))
    K = np.array([[425.,   0., 320.],
       [  0., 425., 256.],
       [  0.,   0.,   1.]])
    camera_matrix = K
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    image_point, _ = cv2.projectPoints(point, rvec, tvec, camera_matrix, dist_coeffs)
    image_point = image_point.squeeze().astype(np.int)
    width = mask.shape[1]
    height = mask.shape[0]
    x = np.clip(image_point[0], 0, width - 1)
    y = np.clip(image_point[1], 0, height - 1)
    return mask[y, x] == True


class GSNet():
    def __init__(self):
        dir = os.path.dirname(os.path.abspath(__file__))
        class Config():
            pass
        self.cfgs = Config()
        self.cfgs.dataset_root = f'{dir}/data/datasets/graspnet'
        self.cfgs.checkpoint_path = f'{dir}/assets/minkuresunet_realsense_tune_epoch20.tar'
        self.cfgs.dump_dir = 'logs'
        self.cfgs.seed_feat_dim = 512
        self.cfgs.camera = 'realsense'
        self.cfgs.num_point = 100000
        self.cfgs.batch_size = 1
        self.cfgs.voxel_size = 0.005
        self.cfgs.collision_thresh = 0.01#
        self.cfgs.voxel_size_cd = 0.01
        self.cfgs.infer = False
        self.cfgs.vis = False
        self.cfgs.scene = '0188'
        self.cfgs.index = '0000'
        
    def inference(self, cloud_masked, max_grasps=5000):
        """Inference grasp from point cloud

        Args:
            cloud_masked (np.ndarray): masked point cloud
            max_grasps (int, optional): max number of grasps to return. Defaults to 200.

        Returns:
            GraspGroup: GraspGroup object
        """
        # sample points random
        if len(cloud_masked) >= self.cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), self.cfgs.num_point, replace=False)
            # print("sampled point cloud idxs:", idxs.shape)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.cfgs.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        data_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                     'coors': cloud_sampled.astype(np.float32) / self.cfgs.voxel_size,
                     'feats': np.ones_like(cloud_sampled).astype(np.float32)}
        
        batch_data = minkowski_collate_fn([data_dict])
        net = GraspNet(seed_feat_dim=self.cfgs.seed_feat_dim, is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        
        # Load checkpoint
        checkpoint = torch.load(self.cfgs.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        # print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

        net.eval()
        tic = time.time()

        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
                
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            if end_points is None:
                return None 
            grasp_preds = pred_decode(end_points)

        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)
        
        # collision detection
        if self.cfgs.collision_thresh > 0:

            cloud = data_dict['point_clouds']

            # Model-free collision detector
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size_cd)
            collision_mask_mfc = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
            gg = gg[~collision_mask_mfc]

            # # Franka collision detector
            # fcdetector = FrankaCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size_cd)
            # collision_mask_fc, global_iou_fc = fcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
            # gg = gg[~collision_mask_fc]
        
        gg = gg.nms()
        gg = gg.sort_by_score()
        
        if gg.__len__() > max_grasps:
            gg = gg[:max_grasps]

        return gg
    
    def visualize(self, cloud, gg: GraspGroup = None, g: Grasp = None, display = True, save_image: str = "output/grasp.png", save_pc: str = "output/grasp.ply"):
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cloud)
        if not display:
            os.environ.pop("DISPLAY", None)
        pcd = cloud
    
        geoms = []
        if gg is not None:
         
            grippers = gg.to_open3d_geometry_list()   
            # o3d.visualization.draw_geometries([pcd, *grippers])
            geoms.extend(grippers)
        elif g is not None:
           
            gripper = g.to_open3d_geometry()
            # o3d.visualization.draw_geometries([pcd, gripper])
            geoms.append(gripper)
        else:
            pass
            # o3d.visualization.draw_geometries([pcd])
        # combined_pcd = o3d.geometry.PointCloud()
        # for geom in geoms:
        
        # # Save the combined point cloud to .ply
        # o3d.io.write_point_cloud(save_pc, combined_pcd)
        # print(f"Saved point cloud to {save_pc}")
        combined_mesh = o3d.geometry.TriangleMesh()
        for geom in geoms:
            if isinstance(geom, o3d.geometry.TriangleMesh):
                combined_mesh += geom

        # Save the combined mesh as a .ply file
        if len(combined_mesh.vertices) > 0:
            o3d.io.write_triangle_mesh(save_pc, combined_mesh)
            print(f"Saved combined mesh to {save_pc}")
        else:
            print("No mesh data to save.")
  
        if display:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=display)
            vis.add_geometry(combined_mesh)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_image)
            print(f"Saved visualization to {save_image}")
            vis.destroy_window()

def visualize_plotly_single(pc, rgb,g: Grasp, extrinsics, max_points=100000, gg_glob=False,rotation=None):
    g_visual = deepcopy(g)
    gg = GraspGroup()
    gg.add(g_visual)
    if gg_glob == False:
        gg.transform(extrinsics)
    gg.rotation_matrices = rotation
    gripper = gg.to_open3d_geometry_list()
    gripper = gripper[0]
    vertices = np.asarray(gripper.vertices)
    triangles = np.asarray(gripper.triangles)
    color = np.asarray(gripper.vertex_colors)[0]
    color = (color * 255).astype(np.uint8)
    color = f'rgb({color[0]}, {color[1]}, {color[2]})'
    grasps_plotly = [go.Mesh3d(
        x=vertices[:, 0], 
        y=vertices[:, 1], 
        z=vertices[:, 2], 
        i=triangles[:, 0], 
        j=triangles[:, 1], 
        k=triangles[:, 2], 
        color=color, 
        opacity=1,
    )]
    pc = pc[:, :3]
    stride = max(1, pc.shape[0] // max_points)
    pc = pc[::stride]
    pc = pc @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    pc_plotly = [go.Scatter3d(
        x=pc[:, 0], 
        y=pc[:, 1], 
        z=pc[:, 2], 
        mode='markers', 
        marker=dict(size=2, color=rgb, opacity=0.8)
        # marker=dict(size=3, color='lightgreen')
    )]
    
    fig = go.Figure(data=pc_plotly + grasps_plotly)
    fig.update_layout(scene_aspectmode='data')
    fig.show()

def visualize_plotly(pc, rgb, gg: GraspGroup, extrinsics, max_points=100000, gg_glob=False):
    """
    Args:
    - pc: np.ndarray[N, 3], point cloud in camera frame
    - gg: GraspGroup, grasps in camera frame
    - extrinsics: np.ndarray[4, 4], extrinsics from camera to table
    - max_points: int, maximum number of points to visualize
    """
    if gg is not None:
        gg_visual = deepcopy(gg)
        if gg_glob == False:
            gg_visual.transform(extrinsics)
        grippers = gg_visual.to_open3d_geometry_list()   

        # visualize grasps
        grasps_plotly = []
        for grasp in grippers:
            vertices = np.asarray(grasp.vertices)
            triangles = np.asarray(grasp.triangles)
            color = np.asarray(grasp.vertex_colors)[0]
            color = (color * 255).astype(np.uint8)
            color = f'rgb({color[0]}, {color[1]}, {color[2]})'
            grasps_plotly.append(go.Mesh3d(
                x=vertices[:, 0], 
                y=vertices[:, 1], 
                z=vertices[:, 2], 
                i=triangles[:, 0], 
                j=triangles[:, 1], 
                k=triangles[:, 2], 
                color=color, 
                opacity=1,
            ))
    else:
        grasps_plotly = []
    # print(extrinsics)
    # visualize pc
    # pc = pc[:, :3]
    # stride = max(1, pc.shape[0] // max_points)
    # pc = pc[::stride]
    # pc = pc @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    # pc_homogeneous = np.hstack([pc, np.ones((pc.shape[0], 1))])
    # pc = np.linalg.inv(extrinsics) @ pc_homogeneous.T
    # pc = pc[:3, :].T
    pc_plotly = [go.Scatter3d(
        x=pc[:, 0], 
        y=pc[:, 1], 
        z=pc[:, 2], 
        mode='markers', 
        marker=dict(size=2, color=200, opacity=0.5)
        # marker=dict(size=3, color='lightgreen')
    )]
    fig = go.Figure(data=pc_plotly + grasps_plotly)
    fig.update_layout(scene_aspectmode='data')
    fig.show()


def rot_matrices2quat(rotations):
    # convert rotation matrices to quaternions
    quats = []
    for rot in rotations:
        quat = R.from_matrix(rot).as_quat()
        quats.append(quat)
    return quats
def quat2rot_matrice(quat):
    
    return R.from_quat(quat).as_matrix()



def grasp_inference(task_description, intrinsic,  object_mask, masked_points, scene_pc_cam, scne_pc, extrinsics, relative_translation_table, relative_rotation_table, save_path1 ="output/waypoints1.npy", save_path2='output/waypoints2.npy', save_path3='output/waypoints3.npy'):

    # masked_points = np.load(pointcloud_path)[:, :3]
    gsnet = GSNet()
    
    gg = gsnet.inference(scene_pc_cam)
    
    if gg is None:
        return None
    
    gg = select_grasp_poses_by_mask(gg, intrinsic, object_mask)
    
    up = np.array([0, 0, 1])   
    gg_valid = GraspGroup()
    for i in range(len(gg)):
        grasp = gg[i]
        mat = extrinsics[:3, :3] @ grasp.rotation_matrix
        z_dir = - mat[:, 0] # mat[:, 2]            
        angle = np.arccos(np.clip(np.dot(z_dir, up), -1.0, 1.0)) 
        
        
        angle_x = np.arccos(np.clip(np.dot(-z_dir, [1,0,0]), -1.0, 1.0))
        
        if angle <= np.deg2rad(55): # and angle_x <= np.deg2rad(90)
            gg_valid.add(grasp)
                
            
    # gg_valid = gg
    num_grasps = len(gg_valid)
    if num_grasps == 0:
        return None, None
    # sort grasps by score
    gg = gg_valid.sort_by_score()

    grasp_clone = GraspGroup()
    for i in range(len(gg)):
        grasp = deepcopy(gg[i]) # gg[i]
        # grasp.translation[0] -= 0.02
        if "drawer" in task_description:
            grasp.translation[2] += 0.04
        else:
            # grasp.translation[2] += 0.03
            grasp.translation += (grasp.depth+0.01) * grasp.rotation_matrix[:, 0]
        grasp_clone.add(grasp)
    gg = grasp_clone

    relative_rotation_camera = extrinsics[:3, :3].T @ relative_rotation_table @ extrinsics[:3, :3]
    relative_translation_table =  relative_translation_table @ extrinsics[:3, :3] 
    
    # get goal grasps
    gg_goal = deepcopy(gg)
    gg_goal.rotation_matrices = relative_rotation_camera @ gg_goal.rotation_matrices
    gg_goal.translations = gg_goal.translations + relative_translation_table  # note that this is not an affine transformation
    
    return gg, gg_goal
    
    
    
    
    
def project_point_to_image(pt, K):
    """
    Projects a 3D point onto a 2D image plane using the camera intrinsic matrix.

    Args:
        pt (tuple): A tuple of (X, Y, Z) coordinates representing the 3D point.
        K (numpy.ndarray): A 3x3 camera intrinsic matrix.

    Returns:
        tuple or None: A tuple (u, v) representing the 2D coordinates on the image
        plane if the point is in front of the camera (Z > 0); otherwise, None.
    """

    X, Y, Z = pt
    if Z <= 0:
        return None  
    u = (K[0, 0] * X) / Z + K[0, 2]
    v = (K[1, 1] * Y) / Z + K[1, 2]
    return u, v




def select_grasp_poses_by_mask(gg_group, K, mask):
    """
    Selects grasp poses from a GraspGroup that correspond to object regions in a mask.

    Args:
    - gg_group: list of Grasp objects, each containing a translation attribute representing
      the 3D position of the grasp.
    - K: np.ndarray, the camera intrinsic matrix used to project 3D points to 2D image coordinates.
    - mask: np.ndarray of shape (height, width), a binary mask where True or 1 indicates
      the presence of an object in the corresponding pixel.

    Returns:
    - selected: list of Grasp objects from gg_group that project to points within the object
      region of the mask.
    """
    selected = []
    height, width = mask.shape  
    for i in range(len(gg_group)):
        grasp = gg_group[i]
        pt = grasp.translation  # [x, y, z]，
        proj = project_point_to_image(pt, K)
        if proj is None:
            continue
        u, v = proj
        u_int = int(round(u))
        v_int = int(round(v))
        if 0 <= u_int < width and 0 <= v_int < height:
            if mask[v_int, u_int]:
                selected.append(grasp)
    return selected
    
