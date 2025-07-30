import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
print(BASE_DIR)
import json
import warnings
import numpy as np
from PIL import Image
from SoFar.depth.utils import depth2pcd, transform_obj_pts
from SoFar.segmentation import sam, grounding_dino as detection
from SoFar.serve.scene_graph import get_scene_graph
from SoFar.serve.utils import generate_rotation_matrix, remove_outliers
from SoFar.serve.pointso import get_model as get_pointofm_model
from SoFar.serve.chatgpt import manip_parsing, manip_spatial_reasoning
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

def sofar(image, depth, intrinsic_matrix, extrinsic_matrix, prompt):

    output_folder = "output"
    
    
    image = Image.fromarray(image)
    image.save("output/img_simpler.png")
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    intrinsic = [fx, fy, cx, cy]
    pcd_camera, pcd_base = depth2pcd(depth, intrinsic, extrinsic_matrix)

    print("\nStart object parsing...")
    info = manip_parsing(prompt, image)
    print(json.dumps(info, indent=2))
    object_list = list(info.keys())

    print("Start Segment Anything...")
    detection_model = detection.get_model()
    sam_model = sam.get_model()
    detections = detection.get_detections(image, object_list, detection_model, output_folder=output_folder)
    mask, ann_img, object_names = sam.get_mask(
        image, object_list, sam_model, detections, output_folder=output_folder)

    print("Generate scene graph...")
    orientation_model = get_pointofm_model()
    objects_info, objects_dict = get_scene_graph(image, pcd_base, mask, info, object_names, orientation_model,
                                                 output_folder=output_folder)
    if len(objects_dict) <= 0:
        raise Exception("No object detected")
    print("objects info:")
    for node in objects_info:
        print(node)
    
    # fsd/er1 pipeline start:
    from waypoint_er1 import get_waypoints
    import numpy as np
    import cv2
    import time

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    ts_str = time.strftime('%Y%m%d_%H%M%S')

    os.makedirs("results/cv_results", exist_ok=True)
    cv2.imwrite(f"results/cv_results/{ts_str} {prompt}.png", image_cv)

    if "put carrot on plate" in prompt:
        pick_sub_task = "Grasp the carrot"
        place_sub_task = prompt
    elif "put eggplant into yellow basket" in prompt:
        pick_sub_task = "Grasp the eggplant"
        place_sub_task = prompt
    elif "stack the green block on the yellow block" in prompt:
        pick_sub_task = "Grasp the green block"
        place_sub_task = prompt
    elif "put the spoon on the towel" in prompt:
        pick_sub_task = "Grasp the spoon"
        place_sub_task = prompt
    else:
        pick_sub_task = "Select grasp point of: " + prompt + "."
        place_sub_task = "Select place point of: " + prompt + "."

    (u, v), points = get_waypoints(np.array(image), pick_sub_task, "pick")

    for point in points:
        image_cv = cv2.circle(image_cv, (int(point[0]), int(point[1])), 5, (0, 0, 127), -1)
    image_cv = cv2.circle(image_cv, (int(u), int(v)), 5, (0, 0, 255), -1)
    pick_goal_uvd = (u, v, depth[int(v), int(u)].item())
    
    (u, v), points = get_waypoints(np.array(image), place_sub_task, "place")
    
    for point in points:
        image_cv = cv2.circle(image_cv, (int(point[0]), int(point[1])), 5, (127, 0, 0), -1)
    image_cv = cv2.circle(image_cv, (int(u), int(v)), 5, (255, 0, 0), -1)
    place_goal_uvd = (u, v, depth[int(v), int(u)].item())
    
    cv2.imwrite(f"results/cv_results/{ts_str} {prompt} result.png", image_cv)
    # fsd/er1 pipeline stop
    
    # match object with pick_goal
    
    def xyz_from_uvd(uvd, intrinsic_matrix, depth_scale):
        u, v, d = uvd
        # https://www.open3d.org/docs/0.6.0/python_api/open3d.geometry.create_point_cloud_from_rgbd_image.html
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        z = d / depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return (x, y, z)

    def uvd_from_xyz(xyz, intrinsic_matrix, depth_scale):
        x, y, z = xyz
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        u = (x * fx / z) + cx
        v = (y * fy / z) + cy
        d = z * depth_scale
        return (u, v, d)
    
    distances = []
    for obj in objects_dict:
        center2base = np.eye(4)
        center2base[:3, 3] = obj["center"]
        # center2cam = base2cam @ xyz2base
        center2cam = np.linalg.inv(extrinsic_matrix) @ center2base
        center_uvd = uvd_from_xyz(center2cam[:3, 3], intrinsic_matrix, 1)
        # image_cv2 = cv2.circle(image_cv, (int(center_uvd[0]), int(center_uvd[1])), 5, (0, 255, 0), -1)
        # cv2.imwrite(f"results/cv_results/{ts_str} {prompt} result_center.png", image_cv2)
        
        distance = np.linalg.norm(np.array(center_uvd)[:2] - np.array(pick_goal_uvd)[:2])
        distances.append(distance)
    
    # xyz2base = cam2base @ xyz2cam
    place_goal_xyz = xyz_from_uvd(place_goal_uvd, intrinsic_matrix, 1)
    place_goal_xyz2cam = np.eye(4)
    place_goal_xyz2cam[:3, 3] = place_goal_xyz
    place_goal_xyz2base = extrinsic_matrix @ place_goal_xyz2cam
    
    image = np.array(image)
    interact_object_id = np.argmin(distances)    
    object_mask = mask[interact_object_id]
    # object_mask = mask[0]
    # segmented_object = pcd[object_mask]
    segmented_object = pcd_camera[object_mask]
    
    obj_pts_base = transform_obj_pts(segmented_object,extrinsic_matrix)
    
    segmented_image = image[object_mask]
    colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
    colored_object_pcd = remove_outliers(colored_object_pcd)
    np.save(os.path.join(output_folder, f"picked_obj.npy"), colored_object_pcd)
    

    
    interact_object_dict = objects_dict[interact_object_id]
    init_position = interact_object_dict["center"]
    target_position = place_goal_xyz2base[:3, 3]
    init_orientation = interact_object_dict["orientation"]
    target_orientation = {}
    
    
    
    
    
    
    
    
    
    # print("Start spatial reasoning...")
    # response = manip_spatial_reasoning(image, prompt, objects_info)
    # import ipdb; ipdb.set_trace()
    # print(response)

    # image = np.array(image)
    # interact_object_id = response["interact_object_id"] - 1
    # object_mask = mask[interact_object_id]
    
    # # object_mask = mask[0]
    # # segmented_object = pcd[object_mask]
    # segmented_object = pcd_camera[object_mask]
    
    # obj_pts_base = transform_obj_pts(segmented_object,extrinsic_matrix)
    
    # segmented_image = image[object_mask]
    # colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
    # colored_object_pcd = remove_outliers(colored_object_pcd)
    # np.save(os.path.join(output_folder, f"picked_obj.npy"), colored_object_pcd)
    

    
    # interact_object_dict = objects_dict[interact_object_id]
    # init_position = interact_object_dict["center"]
    # target_position = response["target_position"]
    # init_orientation = interact_object_dict["orientation"]
    # target_orientation = response["target_orientation"]
    
    
    if len(target_orientation) > 0 and target_orientation.keys() == init_orientation.keys():
        direction_attributes = target_orientation.keys()
        init_directions = [init_orientation[direction] for direction in direction_attributes]
        target_directions = [target_orientation[direction] for direction in direction_attributes]
        transform_matrix = generate_rotation_matrix(np.array(init_directions), np.array(target_directions)).tolist()
    else:
        transform_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    
    result = {
        'init_position': init_position,
        'target_position': target_position,
        'delta_position': [round(target_position[i] - init_position[i], 2) for i in range(3)],
        'init_orientation': init_orientation,
        'target_orientation': target_orientation,
        'transform_matrix': transform_matrix
    }
    print("Result:", result)
    
    return pcd_camera.reshape(-1,3), pcd_base.reshape(-1,3), colored_object_pcd[:,:3], obj_pts_base, object_mask, result['delta_position'], transform_matrix
