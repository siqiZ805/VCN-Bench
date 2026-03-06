import os
import json
import numpy as np
import logging
import random
import torch
import habitat_sim
import quaternion
from quaternion import as_float_array
import supervision as sv
import logging
from collections import Counter
from typing import List, Optional, Tuple, Dict, Union
import copy
import scipy.ndimage as ndimage

from habitat_sim.utils.common import (
    quat_to_coeffs,
    quat_from_angle_axis,
    quat_from_two_vectors,
)
from src.habitat import (
    make_semantic_cfg,
    get_quaternion,
    get_navigable_point_to,
)
from src.geom import get_cam_intr, IoU
from src.utils import rgba2rgb
from src.tsdf_planner import SnapShot
from src.hierarchy_clustering import SceneHierarchicalClustering
from src.habitat import pos_normal_to_habitat, pos_habitat_to_normal

# Local application/library specific imports
from src.conceptgraph.utils.ious import mask_subtract_contained
from src.conceptgraph.utils.general_utils import (
    ObjectClasses,
    measure_time,
    filter_detections,
)
from src.conceptgraph.slam.slam_classes import MapObjectDict, DetectionDict, to_tensor
from src.conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    denoise_objects,
    merge_objects,
    detections_to_obj_pcd_and_bbox,
    processing_needed,
    resize_gobs,
    merge_obj2_into_obj1,
)
from src.conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
)
from src.conceptgraph.utils.model_utils import compute_clip_features_batched


from scipy.spatial.transform import Rotation as R
def camera_to_world(points_cam, cam_pos, cam_rot_quat):
    # R.from_quat输入是(x,y,z,w)
    rot_mat_c2w = R.from_quat(cam_rot_quat).as_matrix()  # 相机→世界的旋转矩阵
    world_coords = (rot_mat_c2w @ points_cam.T).T + cam_pos
    return world_coords


class Scene:
    def __init__(
        self,
        scene_id,
        cfg,
        graph_cfg,
        # detection_model,
        # sam_predictor,
        # clip_model,
        # clip_preprocess,
        # clip_tokenizer,
    ):
        self.cfg = cfg
        # concept graph configuration
        self.cfg_cg = graph_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scene_id = scene_id

        # about the loading the scene
        MP3D_SCENE = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/scene_datasets/mp3d"
        scene_mesh_path = os.path.join(
            MP3D_SCENE, scene_id, scene_id + ".glb"
        )
        navmesh_path = os.path.join(
            MP3D_SCENE, scene_id, scene_id + ".navmesh"
        )
        # semantic_texture_path = os.path.join(
        #     split_path, scene_id, scene_id.split("-")[1] + ".semantic.glb"
        # )
        # scene_semantic_annotation_path = os.path.join(
        #     split_path, scene_id, scene_id.split("-")[1] + ".semantic.txt"
        # )
        assert os.path.exists(
            scene_mesh_path
        ), f"scene_mesh_path: {scene_mesh_path} does not exist"
        assert os.path.exists(
            navmesh_path
        ), f"navmesh_path: {navmesh_path} does not exist"
        # assert os.path.exists(
        #     semantic_texture_path
        # ), f"semantic_texture_path: {semantic_texture_path} does not exist"
        # assert os.path.exists(
        #     scene_semantic_annotation_path
        # ), f"scene_semantic_annotation_path: {scene_semantic_annotation_path} does not exist"
        
        self.load_regions_and_objects()
        self.skip_objects = ['ceiling', 'floor', 'wall', 'object']

        sim_settings = {
            "scene": scene_mesh_path,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": cfg.img_width,
            "height": cfg.img_height,
            "hfov": cfg.hfov,
            "scene_dataset_config_file": cfg.scene_dataset_config_path,
            "camera_tilt": cfg.camera_tilt_deg * np.pi / 180,
        }
        sim_cfg = make_semantic_cfg(sim_settings)
        self.simulator = habitat_sim.Simulator(sim_cfg)
        self.pathfinder = self.simulator.pathfinder
        self.pathfinder.seed(cfg.seed)
        self.pathfinder.load_nav_mesh(navmesh_path)

        # load object classes
        # maintain a list of object classes
        # self.obj_classes = ObjectClasses(
        #     classes_file_path=scene_semantic_annotation_path,
        #     bg_classes=self.cfg_cg["bg_classes"],
        #     skip_bg=self.cfg_cg["skip_bg"],
        #     class_set=self.cfg["class_set"],
        # )

        logging.info(f"Load scene {scene_id} successfully")

        # set agent
        self.agent = self.simulator.initialize_agent(sim_settings["default_agent"])

        self.cam_intrinsic = get_cam_intr(cfg.hfov, cfg.img_width, cfg.img_height)  # K

        # about scene graph
        # self.objects: MapObjectDict[int, Dict] = (
        #     MapObjectDict()
        # )  # object_id -> object item
        # self.object_id_counter = 1
        self.objects = {}  #? {instance_id: {category, object_id, num_detections, bbox} } e.g. "1mp3d_0000_region0/object_25"

        self.snapshots: Dict[str, SnapShot] = {}  # image_path -> snapshot
        self.frames: Dict[str, SnapShot] = {}  # image_path -> all frames
        self.all_observations: Dict[str, np.ndarray] = (
            {}
        )  # image_path -> image, stores all actual observations at each step, used for querying vlm

        self.clustering = SceneHierarchicalClustering(
            min_sample_split=0,
            random_state=66,
        )

        # # setup detection and segmentation models
        # self.detection_model = detection_model
        # self.detection_model.set_classes(self.obj_classes.get_classes_arr())

        # self.sam_predictor = sam_predictor

        # self.clip_model = clip_model.to(self.device)
        # self.clip_preprocess = clip_preprocess
        # self.clip_tokenizer = clip_tokenizer
        
    def load_regions_and_objects(self):
        region_file = "/mnt/hwfile/zhangsiqi1/project/vlfm/data/scene_floors_sim_mp3d_all.json"
        all_regions = json.load(open(region_file, 'r'))
        self.regions = []
        for floor, finfo in all_regions[self.scene_id].items():
            self.regions += finfo['regions']
            
        # embodiedscan_info = "/mnt/petrelfs/zhangsiqi1/efm_data/data/MMScan/embodiedscan_infos_train_val_test.pkl"
        # scan_data = np.load(embodiedscan_info, allow_pickle=True)
        # self.object_data = {region: info for region, info in scan_data.items()
        #                     if 'matterport3d' in region
        #                     and self.scene_id in region}
        # region e.g.: 'matterport3d/ZMojNkEp431/region0'
        obj_dir = "/mnt/inspurfs/efm_t/huangwensi/vl_ln/vln_llava_data/scene_summary"
        self.object_data = json.load(open(
            os.path.join(obj_dir, self.scene_id, 'object_dict.json'), 'r'
        ))
        # key e.g.: "1mp3d_0000_region0/cabinet_17"
        
    def get_cam_pos_rot(self):
        sensor_state = self.agent.get_state().sensor_states['color_sensor']
        cam_pos = sensor_state.position
        cam_rot = sensor_state.rotation
        cam_rot = np.array(cam_rot.imag.tolist() + [cam_rot.real])
        return cam_pos, cam_rot
        
    def pixel_to_point(self, pixel, depth, cam_pos, cam_rot):
        K = self.cam_intrinsic
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        
        u, v = pixel
        cam_z = -depth[v, u]
        cam_x = (cx-u)/fx * cam_z
        cam_y = (v-cy) / fy * cam_z
        cam_coord = np.array([cam_x, cam_y, cam_z])
        world_pos = camera_to_world(cam_coord, cam_pos, cam_rot)
        return world_pos
        

    def __del__(self):
        try:
            self.simulator.close()
        except:
            pass

    def clear_up_detections(self):
        # self.objects = MapObjectDict()
        self.objects = {}
        self.object_id_counter = 1

        self.snapshots = {}
        self.frames = {}
        self.all_observations = {}

    def get_observation(self, pts, angle=None, rotation=None):
        assert (angle is None) ^ (
            rotation is None
        ), "Only one of angle and rotation should be specified"

        agent_state = habitat_sim.AgentState()
        agent_state.position = pts
        if angle is not None:
            agent_state.rotation = get_quaternion(angle, 0)
        else:
            agent_state.rotation = rotation
        self.agent.set_state(agent_state)

        obs = self.simulator.get_sensor_observations()

        # get camera extrinsic matrix
        sensor = self.agent.get_state().sensor_states["depth_sensor"]
        quaternion_0 = sensor.rotation
        translation_0 = sensor.position
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
        cam_pose[:3, 3] = translation_0

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs, cam_pose

    def get_frontier_observation(self, pts, view_dir, camera_tilt=0.0):
        agent_state = habitat_sim.AgentState()

        # solve edge cases of viewing direction
        default_view_dir = np.asarray([0.0, 0.0, -1.0])
        if np.linalg.norm(view_dir) < 1e-3:
            view_dir = default_view_dir
        view_dir = view_dir / np.linalg.norm(view_dir)

        agent_state.position = pts
        # set agent observation direction
        if np.dot(view_dir, default_view_dir) / np.linalg.norm(view_dir) < -1 + 1e-3:
            # if the rotation is to rotate 180 degree, then the quaternion is not unique
            # we need to specify rotating along y-axis
            agent_state.rotation = quat_to_coeffs(
                quaternion.quaternion(0, 0, 1, 0)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()
        else:
            agent_state.rotation = quat_to_coeffs(
                quat_from_two_vectors(default_view_dir, view_dir)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()

        self.agent.set_state(agent_state)
        obs = self.simulator.get_sensor_observations()

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs

    def get_frontier_observation_and_detect_target(
        self,
        pts,
        view_dir,
        detection_model,
        target_obj_id,
        target_obj_class,
        camera_tilt=0.0,
    ):
        obs = self.get_frontier_observation(pts, view_dir, camera_tilt)

        # detect target object
        rgb = obs["color_sensor"]
        semantic_obs = obs["semantic_sensor"]

        detection_model.set_classes([target_obj_class])
        results = detection_model.infer(
            rgb[..., :3], confidence=self.cfg.scene_graph.confidence
        )
        detections = sv.Detections.from_inference(results).with_nms(
            threshold=self.cfg.scene_graph.nms_threshold
        )

        target_detected = False
        if target_obj_id in np.unique(semantic_obs):
            for i in range(len(detections)):
                x_start, y_start, x_end, y_end = detections.xyxy[i].astype(int)
                bbox_mask = np.zeros(semantic_obs.shape, dtype=bool)
                bbox_mask[y_start:y_end, x_start:x_end] = True

                target_x_start, target_y_start = np.argwhere(
                    semantic_obs == target_obj_id
                ).min(axis=0)
                target_x_end, target_y_end = np.argwhere(
                    semantic_obs == target_obj_id
                ).max(axis=0)
                obj_mask = np.zeros(semantic_obs.shape, dtype=bool)
                obj_mask[target_x_start:target_x_end, target_y_start:target_y_end] = (
                    True
                )
                if IoU(bbox_mask, obj_mask) > self.cfg.scene_graph.iou_threshold:
                    target_detected = True
                    break

        return obs, target_detected

    def get_navigable_point_to(
        self,
        target_position,
        max_search=1000,
        min_dist=6.0,
        max_dist=999.0,
        prev_start_positions=None,
    ):
        self.pathfinder.seed(random.randint(0, 1000000))
        return get_navigable_point_to(
            target_position,
            self.pathfinder,
            max_search,
            min_dist,
            max_dist,
            prev_start_positions,
        )
        
    def update_scene_graph(self, pts, tsdf, img_path, ):
        '''
        pts is habitat
        '''
        rotation = self.agent.get_state().rotation
        rotation = rotation.imag.tolist() + [rotation.real]
        frame = SnapShot(
            image=img_path, color=(random.random(), random.random(), random.random()),
            obs_point=tsdf.habitat2voxel(pts), 
            agent_position=pts, agent_rotation=rotation
        )
        added_obj_ids = []
        
        normal_pts = pos_habitat_to_normal(pts)
        _, unocc = tsdf.get_island_around_pts(normal_pts, height=1.8)
        obstacle_map = tsdf.get_obstacle_map(height=1.8)
        # convolution to get the obstacles together with surroundings
        kernel_size = int(0.5 / tsdf._voxel_size)
        kernel = np.ones((kernel_size, kernel_size))
        obstacle_map_convolved = ndimage.convolve(
            obstacle_map.astype(float), kernel, mode="constant", cval=0.0
        )
        
        cand_objs = self.filter_obj_with_distance(pts)
        for objinfo in cand_objs:
            objid = objinfo['instance_id']
            obj_pixel = tsdf.habitat2voxel(objinfo['position'])
            x, y = obj_pixel[:2]
            if unocc[x,y] > 0 or obstacle_map_convolved[x,y] > 0:  #? 物体可能在障碍物上
                frame.full_obj_list[objid] = 1  # objid: confidence
                if objid not in self.objects:
                    added_obj_ids.append(objid)
                    self.objects[objid] = {
                        'category': objinfo['category'],
                        'object_id': objid.split('_')[-1],
                        'bbox': {
                            'center': objinfo['position'],
                            'min_point': objinfo['min_points'][:2] + objinfo['max_points'][2:],
                            'max_point': objinfo['max_points'][:2] + objinfo['min_points'][2:]
                        },
                        'num_detections': 1
                    }
        self.frames[img_path] = frame
        return added_obj_ids
        
    def get_region_from_point(self, pts, eps=0.1):
        '''
        pts is habitat
        '''
        return_regions = []
        for region in self.regions:
            center = region['region_center']
            sizes = region['region_sizes']
            x_min = center[0] - sizes[0] / 2.0
            y_min = center[1] - sizes[1] / 2.0
            z_min = center[2] - sizes[2] / 2.0
            x_max = center[0] + sizes[0] / 2.0
            y_max = center[1] + sizes[1] / 2.0
            z_max = center[2] + sizes[2] / 2.0
            
            if (x_min - eps <= pts[0] <= x_max + eps) and \
                (y_min - eps <= pts[1] <= y_max + eps) and \
                (z_min - eps <= pts[2] <= z_max + eps):
                return_regions.append(region['region_id'])
        return return_regions

    def filter_obj_with_distance(self, pts):
        '''
        pts is habitat
        '''
        cand_regions = self.get_region_from_point(pts)
        cand_floors = [x.split('_')[0] for x in cand_regions]
        cand_floors = list(set(cand_floors))
        if len(cand_floors) == 1:
            floor = cand_floors[0]
            floor_regions = [x for x in self.regions
                if x['region_id'].startswith(f"{floor}_")]
        elif len(cand_floors) > 1:
            floor_regions = [x for x in self.regions
                if x['region_id'].split('_')[0] in cand_floors]
        else:
            floor_regions = self.regions
        
        floor_region_ids = [x['region_id'].split('_')[-1] for x in floor_regions]
        
        cand_objs = []
        for objid, objinfo in self.object_data.items():
            region_id = objid.split('/')[0].split('region')[-1]
            if region_id not in floor_region_ids:
                continue
            if objinfo['category'] in self.skip_objects:
                continue
            
            center = objinfo['position']
            min_point = objinfo['min_points'][:2] + objinfo['max_points'][2:]
            max_point = objinfo['max_points'][:2] + objinfo['min_points'][2:]
            
            flag = False
            for point in [center, min_point, max_point]:
                xy_dist = np.sqrt((pts[0]-point[0])**2 + (pts[2]-point[2])**2)
                if xy_dist < 3.5 and -1 < point[1] - pts[1] < 3:
                    flag = True
                    break
            if flag:
                cand_objs.append(objinfo)
        return cand_objs


    def filter_gobs_with_distance_ori(self, pts, gobs):
        idx_to_keep = []
        for idx in range(len(gobs["bbox"])):
            if gobs["bbox"][idx] is None:  # point cloud was discarded
                continue

            # get the distance between the object and the current observation point
            if (
                np.linalg.norm(gobs["bbox"][idx].center[[0, 2]] - pts[[0, 2]])
                > self.cfg.scene_graph.obj_include_dist
            ):
                logging.debug(
                    f"Object {gobs['detection_class_labels'][idx]} is too far away, skipping"
                )
                continue
            idx_to_keep.append(idx)

        for attribute in gobs.keys():
            if isinstance(gobs[attribute], str) or attribute == "classes":  # Captions
                continue
            if attribute in ["labels", "edges", "text_feats", "captions"]:
                # Note: this statement was used to also exempt 'detection_class_labels' but that causes a bug. It causes the edges to be misalgined with the objects.
                continue
            elif isinstance(gobs[attribute], list):
                gobs[attribute] = [gobs[attribute][i] for i in idx_to_keep]
            elif isinstance(gobs[attribute], np.ndarray):
                gobs[attribute] = gobs[attribute][idx_to_keep]
            else:
                raise NotImplementedError(f"Unhandled type {type(gobs[attribute])}")

        return gobs

    def merge_obj_matches(
        self,
        detection_list: DetectionDict,
        match_indices: List[Tuple[int, Optional[int]]],
        obj_classes: ObjectClasses,
        snapshot: SnapShot,
        target_obj_id_mapping: Dict[int, int],
    ) -> Tuple[List[str], Dict[int, int], List[int], List[int]]:
        visualize_captions = []
        all_obj_ids = []
        added_obj_ids = []
        for idx, (detected_obj_id, existing_obj_match_id) in enumerate(match_indices):
            if existing_obj_match_id is None:
                self.objects[detected_obj_id] = detection_list[detected_obj_id]
                visualize_captions.append(
                    f"{detected_obj_id} {self.objects[detected_obj_id]['class_name']} {self.objects[detected_obj_id]['conf']:.3f} N"
                )
                all_obj_ids.append(detected_obj_id)
                added_obj_ids.append(detected_obj_id)
            else:
                # merge detected object into existing object
                detected_obj = detection_list[detected_obj_id]
                matched_obj = self.objects[existing_obj_match_id]

                merged_obj = merge_obj2_into_obj1(
                    obj1=matched_obj,
                    obj2=detected_obj,
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                    spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                    device=self.device,
                    run_dbscan=False,
                )
                # fix the class name by adopting the most popular class name
                class_id_counter = Counter(merged_obj["class_id"])
                most_common_class_id = class_id_counter.most_common(1)[0][0]
                most_common_class_name = obj_classes.get_classes_arr()[
                    most_common_class_id
                ]
                merged_obj["class_name"] = most_common_class_name

                # adjust the full detected list of the current snapshot: remove the detected object and add the merged object
                snapshot.full_obj_list[existing_obj_match_id] = detected_obj["conf"]
                snapshot.full_obj_list.pop(detected_obj_id)

                self.objects[existing_obj_match_id] = merged_obj
                visualize_captions.append(
                    f"{existing_obj_match_id} {self.objects[existing_obj_match_id]['class_name']} {detected_obj['conf']:.3f} {merged_obj['num_detections']}"
                )
                all_obj_ids.append(existing_obj_match_id)

                # update the mapping of target object id
                for gt_id, mapped_id in target_obj_id_mapping.items():
                    if mapped_id == detected_obj_id:
                        target_obj_id_mapping[gt_id] = existing_obj_match_id

        return visualize_captions, target_obj_id_mapping, added_obj_ids, all_obj_ids

    def make_detection_list_from_pcd_and_gobs(
        self, gobs, image_path, obj_classes
    ) -> DetectionDict:
        detection_list = DetectionDict()
        for mask_idx in range(len(gobs["mask"])):
            if gobs["pcd"][mask_idx] is None:  # point cloud was discarded
                continue

            curr_class_name = gobs["classes"][gobs["class_id"][mask_idx]]
            curr_class_idx = obj_classes.get_classes_arr().index(curr_class_name)

            detected_object = {
                "id": self.object_id_counter,  # unique id for this object
                "class_name": curr_class_name,  # global class id for this detection
                "class_id": [curr_class_idx],  # global class id for this detection
                "num_detections": 1,  # number of detections in this object
                "conf": gobs["confidence"][mask_idx],
                # These are for the entire 3D object
                "pcd": gobs["pcd"][mask_idx],
                "bbox": gobs["bbox"][mask_idx],
                "clip_ft": to_tensor(gobs["image_feats"][mask_idx]),
                # the snapshot name it belongs to
                "image": None,
            }

            detection_list[self.object_id_counter] = detected_object
            self.object_id_counter += 1

        return detection_list

    def cleanup_empty_frames_snapshots(self):
        # remove the frame that have empty detected objects
        filtered_frames = {}
        for file_name, frame in self.frames.items():
            if len(frame.full_obj_list) > 0:
                filtered_frames[file_name] = frame
        self.frames = filtered_frames

        # remove the snapshots that have no cluster
        filtered_snapshots = {}
        for file_name, snapshot in self.snapshots.items():
            if len(snapshot.cluster) > 0:
                filtered_snapshots[file_name] = snapshot
        self.snapshots = filtered_snapshots

    def update_snapshots(
        self,
        obj_ids,
        min_detection=2,
    ):
        self.cleanup_empty_frames_snapshots()

        prev_snapshots = copy.deepcopy(self.snapshots)

        obj_ids_temp = obj_ids.copy()
        for filename, snapshot in self.snapshots.items():
            cluster = snapshot.cluster
            if any([obj_id in obj_ids_temp for obj_id in cluster]):
                obj_ids = obj_ids.union(set(cluster))
                prev_snapshots.pop(filename)
        obj_ids = list(set(obj_ids))

        # # find and exclude the objects that have only one observation
        # obj_exclude = [
        #     obj_id
        #     for obj_id in self.objects.keys()
        #     if self.objects[obj_id]["num_detections"] < min_detection
        # ]
        # obj_ids = [obj_id for obj_id in obj_ids if obj_id not in obj_exclude]

        obj_centers = np.zeros((len(obj_ids), 2))
        for i, obj_id in enumerate(obj_ids):
            obj_centers[i] = np.array(self.objects[obj_id]["bbox"]['center'])[[0, 2]]

        if len(obj_centers) == 0:
            return

        new_snapshots = self.clustering.fit(obj_centers, obj_ids, self.frames)

        prev_snapshot_objs = [
            obj_id
            for snapshot in prev_snapshots.values()
            for obj_id in snapshot.cluster
        ]
        assert set(
            [
                obj_id
                for snapshot in new_snapshots.values()
                for obj_id in snapshot.cluster
            ]
        ) == set(
            obj_ids
        ), f"{set([obj_id for snapshot in new_snapshots.values() for obj_id in snapshot.cluster])} != {set(obj_ids)}"
        assert (
            set(obj_ids) & set(prev_snapshot_objs)
        ) == set(), f"{set(obj_ids)} & {set(prev_snapshot_objs)} != empty"
        assert (set(obj_ids) | set(prev_snapshot_objs)) == set(
            self.objects.keys()
        ), f"{set(obj_ids)} | {set(prev_snapshot_objs)}  != {set(self.objects.keys())}"
        # | {set(obj_exclude)}

        for key, snapshot in new_snapshots.items():
            if key in prev_snapshots.keys():
                prev_snapshots[key].cluster += snapshot.cluster
            else:
                prev_snapshots[key] = snapshot
        self.snapshots = prev_snapshots

        # update the snapshot belonging of each object
        for file_name, snapshot in self.snapshots.items():
            for obj_id in snapshot.cluster:
                self.objects[obj_id]["image"] = file_name

        # remove the duplicates caused by copying snapshots: self.frames and self.snapshots should point to the same object
        for file_name, snapshot in self.snapshots.items():
            self.frames[file_name] = snapshot

        # sanity check
        for obj_id, obj in self.objects.items():
            if obj["num_detections"] < min_detection:
                assert (
                    obj["image"] is None
                ), f"{obj_id} has only one detection but has image"
            else:
                assert obj["image"] is not None, f"{obj_id} has no image"

    def periodic_cleanup_objects(self, frame_idx, pts, goal_obj_ids_mapping=None):
        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
            self.cfg_cg["denoise_interval"],
            self.cfg_cg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = measure_time(denoise_objects)(
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.device,
                objects=self.objects,
            )

        # Filtering
        if processing_needed(
            self.cfg_cg["filter_interval"],
            self.cfg_cg["run_filter_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = filter_objects(
                obj_min_points=self.cfg_cg["obj_min_points"],
                obj_min_detections=self.cfg_cg["obj_min_detections"],
                min_distance=self.cfg.scene_graph.obj_include_dist,
                objects=self.objects,
                pts=pts,
            )

        # temporarily we do not merge close objects, since handling which snapshot the merged object belongs to is a bit tricky

        # Merging
        if processing_needed(
            self.cfg_cg["merge_interval"],
            self.cfg_cg["run_merge_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = measure_time(merge_objects)(
                merge_overlap_thresh=self.cfg_cg["merge_overlap_thresh"],
                merge_visual_sim_thresh=self.cfg_cg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=self.cfg_cg["merge_text_sim_thresh"],
                objects=self.objects,
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.device,
                goal_obj_ids_mapping=goal_obj_ids_mapping,
            )

        # update the object list in snapshots, since some objects may have been removed
        frame_to_pop = []
        for (
            filename,
            ss,
        ) in (
            self.frames.items()
        ):  # TODO: check whether content in snapshots are also changed, and see whether need to remove snapshot that have empty cluster
            ss.cluster = [
                obj_id for obj_id in ss.cluster if obj_id in self.objects.keys()
            ]
            ss.full_obj_list = {
                obj_id: conf
                for obj_id, conf in ss.full_obj_list.items()
                if obj_id in self.objects.keys()
            }
            if len(ss.full_obj_list) == 0:
                frame_to_pop.append(filename)
        for filename in frame_to_pop:
            self.frames.pop(filename)

        # update the goal object ids mapping to remove the objects that have been removed
        if goal_obj_ids_mapping is not None:
            for goal_obj_id, mapped_obj_ids in goal_obj_ids_mapping.items():
                goal_obj_ids_mapping[goal_obj_id] = [
                    obj_id for obj_id in mapped_obj_ids if obj_id in self.objects.keys()
                ]

    def sanity_check(self, cfg):
        # sanity check
        obj_exclude_count = sum(
            [
                1 if obj["num_detections"] < cfg.min_detection else 0
                for obj in self.objects.values()
            ]
        )
        total_objs_count = sum(
            [len(snapshot.cluster) for snapshot in self.snapshots.values()]
        )
        assert (
            len(self.objects) == total_objs_count + obj_exclude_count
        ), f"{len(self.objects)} != {total_objs_count} + {obj_exclude_count}"
        total_objs_count = sum(
            [len(set(snapshot.cluster)) for snapshot in self.snapshots.values()]
        )
        assert (
            len(self.objects) == total_objs_count + obj_exclude_count
        ), f"{len(self.objects)} != {total_objs_count} + {obj_exclude_count}"
        for obj_id in self.objects.keys():
            exist_count = 0
            for ss in self.snapshots.values():
                if obj_id in ss.cluster:
                    exist_count += 1
            if self.objects[obj_id]["num_detections"] < cfg.min_detection:
                assert (
                    exist_count == 0
                ), f"{exist_count} != 0 for obj_id {obj_id}, {self.objects[obj_id]['class_name']}"
            else:
                assert (
                    exist_count == 1
                ), f"{exist_count} != 1 for obj_id {obj_id}, {self.objects[obj_id]['class_name']}"
        for ss in self.snapshots.values():
            assert len(ss.cluster) == len(
                set(ss.cluster)
            ), f"{ss.cluster} has duplicates"
            assert len(ss.full_obj_list.keys()) == len(
                set(ss.full_obj_list.keys())
            ), f"{ss.full_obj_list.keys()} has duplicates"
            for obj_id in ss.cluster:
                assert (
                    obj_id in ss.full_obj_list
                ), f"{obj_id} not in {ss.full_obj_list.keys()}"
            for obj_id in ss.full_obj_list.keys():
                assert obj_id in self.objects, f"{obj_id} not in scene objects"
        # check whether the snapshots in scene.snapshots and scene.frames are the same
        for file_name, ss in self.snapshots.items():
            assert (
                ss.cluster == self.frames[file_name].cluster
            ), f"{ss}\n!=\n{self.frames[file_name]}"
            assert (
                ss.full_obj_list == self.frames[file_name].full_obj_list
            ), f"{ss}\n==\n{self.frames[file_name]}"

    def print_scene_graph(self):
        snapshot_dict = {}
        for obj_id, obj in self.objects.items():
            if obj["image"] not in snapshot_dict:
                snapshot_dict[obj["image"]] = []
            snapshot_dict[obj["image"]].append(
                f"{obj_id}: {obj['class_name']} {obj['num_detections']}"
            )
        for snapshot_id, obj_list in snapshot_dict.items():
            logging.info(f"{snapshot_id}:")
            for obj_str in obj_list:
                logging.info(f"\t{obj_str}")
