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

sys.path.append("/home/chq/detectron2")

from detectron2.utils.visualizer import GenericMask
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# copy from mocha project

def mask_to_polygon(mask):
    h, w = mask.shape[:2]
    gm = GenericMask(mask, h, w)
    vertices = []
    for p in gm.polygons:
        xy = p.reshape(-1, 2)
        vertices.append(xy)
    vertices = np.concatenate(vertices, axis=0)
    return vertices

# farthest point sampling
def fps(points, n_samples):
    """
    points: [N, 2] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype="int")  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float("inf")  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
            (points[last_added] - points[points_left]) ** 2
        ).sum(
            -1
        )  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(
            dist_to_last_added_point, dists[points_left]
        )  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]

def get_vertices(mask): # from get_segmentation_masks
    vertices = mask_to_polygon(mask)

    center_point = vertices.mean(0)
    vertices = np.concatenate([center_point[None, ...], vertices], axis=0) # including center point
    
    num_samples = 5
    if vertices.shape[0] > num_samples:
        kps = fps(vertices, num_samples)
    else:
        kps = vertices # [N, 2]

    kps = np.concatenate([kps[[0]], kps[1:][kps[1:, 1].argsort()]], axis=0) # sorting
    vertices = kps
    return vertices
    
def load_prompts():
    """Load prompts from files.
    """
    prompts = dict()
    prompt_dir = './prompts/visual_prompt_planner'
    
    for filename in os.listdir(prompt_dir):
        path = os.path.join(prompt_dir, filename)
        if os.path.isfile(path) and path[-4:] == '.txt':
            with open(path, 'r') as f:
                value = f.read()
            key = filename[:-4]
            prompts[key] = value
    return prompts
    
prompts = load_prompts()

def remove_trailing_comments(input_string):
    # Split the input string into lines
    lines = input_string.split('\n')

    # Process each line to remove comments
    processed_lines = []
    for line in lines:
        comment_index = line.find('//')
        if comment_index != -1:
            # Remove the comment
            line = line[:comment_index]
        processed_lines.append(line.strip())

    # Join the processed lines back into a single string
    return """{}""".format('\n'.join(processed_lines))

def parse_json_string(res, verbose=False):
    if '```' in res:
        try:
            res_clean = res

            if '```json' in res:
                res_clean = res_clean.split('```')[1].split('json')[1]
            elif '```JSON' in res:
                res_clean = res_clean.split('```')[1].split('JSON')[1]
            elif '```' in res:
                res_clean = res_clean.split('```')[1]
            else:
                print('Invalid response: ')
                print(res)

        except Exception:
            import traceback
            print(traceback.format_exc())
            print('Invalid response: ')
            print(res)
            return None
    else:
        res_clean = res

    try:
        res_filtered = remove_trailing_comments(res_clean)
        object_info = json.loads(res_filtered)

        # if verbose:
        #     print_object_info(object_info)

        return object_info

    except Exception:
        import traceback
        print(traceback.format_exc())
        print('The original response: ')
        print(res)
        print('Invalid cleaned response: ')
        print(res_clean)
        return None







from string import ascii_lowercase
import io

def plot_keypoints(
        ax, image_size, keypoints, color, prefix='', annotate_index=True,
        add_caption=True):
    if keypoints is None:
        return

    (h, w) = image_size
    for i, keypoint in enumerate(keypoints):
        if keypoint is None:
            continue

        ax.plot(
            keypoint[0], keypoint[1],
            color=color, alpha=0.4,
            marker='o', markersize=10,
            markeredgewidth=2, markeredgecolor='black',
        )

        if add_caption:
            text = ''
            if annotate_index:
                text = text + str(i + 1)
            text = prefix + text

            xytext = (
                min(max(30, keypoint[0]), w - 30),
                min(max(30, keypoint[1]), h - 30),
            )

            ax.annotate(text, keypoint, xytext, size=12,)

def annotate_candidate_keypoints(
        image,
        candidate_keypoints,
):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')

    image_size = image.size[:2]
    print('image_size (annotate_candidate_keypoints)', image_size)

    if candidate_keypoints['grasped'] is not None:
        plot_keypoints(
            ax, image_size, candidate_keypoints['grasped'], 'r', prefix='P')

    if candidate_keypoints['unattached'] is not None:
        plot_keypoints(
            ax, image_size, candidate_keypoints['unattached'], 'b', prefix='Q')

    buf = io.BytesIO()
    fig.savefig(buf, transparent=True, bbox_inches='tight',
                pad_inches=0, format='jpg')
    buf.seek(0)
    # close the figure to prevent it from being displayed
    plt.close(fig)
    return Image.open(buf)

def annotate_grid(image, grid_size):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')

    image_size = image.size[:2]
    (w, h) = image_size

    for i in range(1, grid_size[0]):
        ax.hlines(h * i / grid_size[0], 0, w,
                  color='black', alpha=0.3, linewidth=1)

    for j in range(1, grid_size[1]):
        ax.vlines(w * j / grid_size[1], 0, h,
                  color='black', alpha=0.3, linewidth=1)

    # for i in range(0, grid_size[0]):
    #     ax.annotate(str(i + 1),
    #                 [w * (i + 0.5) / grid_size[0], 0],
    #                 [w * (i + 0.5) / grid_size[0], -10],
    #                 size=12)

    # for i in range(0, grid_size[0]):
    #     ax.annotate(ascii_lowercase[i],
    #                 [0, h * (i + 0.5) / grid_size[0]],
    #                 [-20, h * (i + 0.5) / grid_size[0]],
    #                 size=12)

    for i in range(0, grid_size[0]):
        for j in range(0, grid_size[1]):
            ax.annotate(str(f"{ascii_lowercase[i]}{grid_size[1] - j}"),
                        [w * (i + 0.5) / grid_size[0], h *
                         (j + 0.5) / grid_size[1]],
                        [w * (i + 0.5) / grid_size[0], h *
                         (j + 0.5) / grid_size[1]],
                        size=10,
                        color='white')

    buf = io.BytesIO()
    fig.savefig(buf, transparent=True, bbox_inches='tight',
                pad_inches=0, format='jpg')
    buf.seek(0)
    # close the figure to prevent it from being displayed
    plt.close(fig)
    return Image.open(buf)

def annotate_visual_prompts(
        obs_image,
        candidate_keypoints,
        waypoint_grid_size,
        log_dir=None,
):
    """Annotate the visual prompts on the image.
    """
    annotated_image = annotate_candidate_keypoints(
        obs_image,
        candidate_keypoints,
    )
    if log_dir is not None:
        annotated_image.save(os.path.join(log_dir, 'keypoints.png'))
    annotated_image = annotate_grid(
        annotated_image,
        waypoint_grid_size,
    )
    if log_dir is not None:
        annotated_image.save(os.path.join(log_dir, 'grid.png'))
    return annotated_image

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
        
    # import ipdb; ipdb.set_trace()

    from gpt_utils import request_gpt
    
    text_requests = [f"Task: {prompt}", " and you need make sure object name in this list: " + str(object_names)]
    image_requests = [image]
    
    res = request_gpt(
        text_requests,
        image_requests,
        prompts['propose_subtasks'],
    )

    # Merge picking and placing into higher-evel subtasks.
    res_filtered = request_gpt(
        text_requests + [res],
        image_requests,
        prompts['filter_subtasks'],
    )

    plan_info = parse_json_string(res_filtered)
    
    plan_info = plan_info[0]
    
    candidate_keypoints = {}
    
    for msk, obj_name in zip(mask, object_names):
        if obj_name in plan_info['object_grasped']:
            candidate_keypoints['grasped'] = get_vertices(msk)
        
        if obj_name in plan_info['object_unattached']:
            candidate_keypoints['unattached'] = get_vertices(msk)
            
    waypoint_grid_size = [5, 5]
    
    # import ipdb; ipdb.set_trace()

    annotated_image = annotate_visual_prompts(
        image,
        candidate_keypoints,
        waypoint_grid_size=waypoint_grid_size,
        log_dir="./",
    )
    
    text_requests = [f"Task: {plan_info}"]

    ctx = None
    for i in range(3):
        res = request_gpt(
            text_requests,
            [annotated_image],
            prompts['select_motion'])

        ctx = parse_json_string(res)
        
        if ctx is not None:
            break
            
    # import ipdb; ipdb.set_trace()
    
    import cv2
    import time
    
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    ts_str = time.strftime('%Y%m%d_%H%M%S')

    os.makedirs("results/cv_results", exist_ok=True)
    cv2.imwrite(f"results/cv_results/{ts_str} {prompt}.png", image_cv)
        
    val = ctx['grasp_keypoint']
    assert val != ''
    idx = int(val[1:]) - 1
    uu, vv = candidate_keypoints['grasped'][idx]
    
    image_cv = cv2.circle(image_cv, (int(uu), int(vv)), 5, (0, 0, 255), -1)
    pick_goal_uvd = (uu, vv, depth[int(vv), int(uu)].item())

    val = ctx['target_keypoint']
    assert val != ''
    idx = int(val[1:]) - 1
    uu, vv = candidate_keypoints['unattached'][idx]
    
    image_cv = cv2.circle(image_cv, (int(uu), int(vv)), 5, (255, 0, 0), -1)
    place_goal_uvd = (uu, vv, depth[int(vv), int(uu)].item())
    
    cv2.imwrite(f"results/cv_results/{ts_str} {prompt} result.png", image_cv)
    
    # cv2.imwrite("img.png", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("img_mask.png", cv2.cvtColor(np.array(mask[0] * 255, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    
    # debug_image = np.array(image)
    # for (x, y) in vertices: cv2.circle(debug_image, (int(x), int(y)), 5, (0, 0, 255), -1)
    # cv2.imwrite("img_mask_debug.png", cv2.cvtColor(np.array(debug_image), cv2.COLOR_RGB2BGR))
    
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
