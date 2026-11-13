from src.runtime import configure_environment, run_cli, select_episode_range

configure_environment()


def main(cfg, start_ratio=0.0, end_ratio=1.0, split=1):
    """Run navigation evaluation."""
    import gzip
    import json
    import logging
    import math
    import os
    import random
    from pathlib import Path
    from pprint import pprint

    import matplotlib.pyplot as plt
    import numpy as np
    import open_clip
    import torch
    from omegaconf import OmegaConf
    from ultralytics import SAM, YOLOWorld

    from model.query_3dmem import (
        build_model,
        evaluation_identity,
        prepare_vlm_input_dict,
        query_qwen,
        result_is_reusable,
    )
    from src.geom import get_cam_intr, get_scene_bnds
    from src.habitat import pose_habitat_to_tsdf
    from src.logger import Logger
    from src.scene_memo_eval import Scene
    from src.tour_video import sample_tour_video
    from src.tsdf_planner import Frontier, SnapShot, TSDFPlanner
    from src.utils import (
        calc_agent_subtask_distance,
        get_pts_angle_goatbench,
        resize_image,
    )

    identity = evaluation_identity(OmegaConf.to_container(cfg, resolve=True))
    # Reject incompatible cached results before loading any model weights.
    for result_path in Path(cfg.output_dir).glob("*_ep_*/result.json"):
        with result_path.open() as stream:
            result_is_reusable(json.load(stream), identity)

    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    MP3D_SCENE = cfg.mp3d_dir
    scenes = [x for x in os.listdir(MP3D_SCENE) if "." not in x]
    with gzip.open(cfg.test_data_path, "r") as f:
        episodes = select_episode_range(
            json.load(f)["episodes"], start_ratio, end_ratio
        )
    num_episode = len(episodes)
    logging.info(f"Total number of episodes: {num_episode}")

    # load detection and segmentation models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    clip_model, clip_preprocess = open_clip.create_model_from_pretrained(
        "ViT-B-32", cfg.clip_model_path
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")

    model, processor = build_model(cfg.qwen_path, cfg.get("qwen_processor_path"))
    logging.info(f"Load QWEN model successful!")

    # Initialize the logger
    logger = Logger(cfg.output_dir, voxel_size=cfg.tsdf_grid_size)
    scene = None
    for scene_id in scenes:
        scene_data = [x for x in episodes if scene_id in x["scene_id"]]
        if scene_data == []:
            continue
        total_episodes = len(scene_data)
        random.shuffle(scene_data)
        for episode_idx, episode in enumerate(scene_data):

            logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            logging.info(f"Loading scene {scene_id}")
            episode_id = episode["episode_id"]

            # load finished epids
            output_path = os.path.join(
                cfg.output_dir, f"{scene_id}_ep_{episode_id}", "result.json"
            )
            if os.path.exists(output_path):
                with open(output_path) as stream:
                    if result_is_reusable(json.load(stream), identity):
                        continue
                logging.info(
                    "Retrying episode with a previously recorded evaluation error"
                )

            pts_init, angle_init = get_pts_angle_goatbench(
                episode["start_position"], episode["start_rotation"]
            )

            # load prior video
            hkey = episode["hkey"]
            video_dir = os.path.join(cfg.video_root, scene_id, hkey)
            all_rgbs = sorted(os.listdir(video_dir))
            sidx = episode["start_idx"]
            eidx = episode["end_idx"]
            tour_video = all_rgbs[sidx:eidx]
            tour_video = sample_tour_video(tour_video, cfg.video_max_frame)

            # load scene
            if scene is not None:
                scene.close()
            scene = Scene(
                scene_id,
                cfg,
                cfg_cg,
                detection_model,
                sam_predictor,
                clip_model,
                clip_preprocess,
                clip_tokenizer,
            )

            # initialize the TSDF
            floor_height = pts_init[1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            num_step = max(num_step, 50)

            # episode info
            episode_dir, eps_frontier_dir, eps_snapshot_dir = logger.init_episode(
                episode_id=f"{scene_id}_ep_{episode_id}"
            )

            max_attempts = 3
            for attempt in range(max_attempts):
                logging.info(
                    f"Episode {episode_id} attempt {attempt + 1}/{max_attempts}"
                )
                pts = np.array(pts_init, copy=True)
                angle = float(angle_init)
                for artifact_dir in (
                    eps_frontier_dir,
                    eps_snapshot_dir,
                    os.path.join(episode_dir, "visualization"),
                ):
                    if not os.path.isdir(artifact_dir):
                        continue
                    for name in os.listdir(artifact_dir):
                        artifact_path = os.path.join(artifact_dir, name)
                        if os.path.isfile(artifact_path):
                            os.remove(artifact_path)
                target_path = os.path.join(episode_dir, "target.png")
                if os.path.isfile(target_path):
                    os.remove(target_path)
                scene.clear_up_detections()

                tsdf_planner = TSDFPlanner(
                    vol_bnds=tsdf_bnds,
                    voxel_size=cfg.tsdf_grid_size,
                    floor_height=floor_height,
                    floor_height_offset=0,
                    pts_init=pts,
                    init_clearance=cfg.init_clearance * 2,
                    save_visualization=cfg.save_visualization,
                    vision_height=1.5,
                )
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None
                metadata = logger.init_eval_task(episode, pts, tsdf_planner)
                # instruction, class, goal_obj_ids, goal_positions_voxel, viewpoints, gt_dist
                logging.info(f"\n\nScene {scene_id} initialization successful!")
                # mapping from the obj id in habitat to the id assigned by concept graph
                # this mapping/alignment is done by heuristic matching between object masks
                goal_obj_ids_mapping = {
                    obj_id: [] for obj_id in metadata["goal_obj_ids"]
                }
                max_point_choice = None
                target_obj_ids_estimate = []

                # run step
                global_step = -1
                task_success = False
                my_result = {
                    **identity,
                    "model_checkpoint": cfg.qwen_path,
                    "processor_checkpoint": cfg.get("qwen_processor_path")
                    or cfg.qwen_path,
                    "prior_video_frames": tour_video,
                    "goal": metadata["goal"],
                    "evaluation_status": "completed",
                    "failure_reason": None,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                }
                while global_step < num_step - 1:
                    global_step += 1
                    logging.info(f"\n== global step: {global_step} ==")

                    # 1. Observe the surroundings, update the scene graph and occupancy map
                    # 1.1 Determine the viewing angles for the current step
                    # if global_step == 0:
                    #     angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180  # 40
                    #     total_views = 1 + cfg.extra_view_phase_2   # 1+6
                    # else:
                    #     angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180  # 60
                    #     total_views = 1 + cfg.extra_view_phase_1   # 1+2
                    angle_increment = 90 * np.pi / 180
                    total_views = 4
                    all_angles = [
                        angle + angle_increment * (i - total_views // 2)
                        for i in range(total_views)
                    ]
                    # Let the main viewing angle be the last one to avoid potential overwriting problems
                    main_angle = all_angles.pop(total_views // 2)
                    all_angles.append(main_angle)

                    rgb_egocentric_views = []
                    all_added_obj_ids = (
                        []
                    )  # Record all the objects that are newly added in this step
                    for view_idx, ang in enumerate(all_angles):
                        # For each view
                        obs, cam_pose = scene.get_observation(pts, angle=ang)
                        rgb = obs["color_sensor"]
                        depth = obs["depth_sensor"]
                        semantic_obs = obs["semantic_sensor"]

                        # collect all view features
                        obs_file_name = f"{global_step}-view_{view_idx}.png"
                        logging.debug("Observation: %s", obs_file_name)
                        with torch.no_grad():
                            # Concept graph pipeline update
                            annotated_rgb, added_obj_ids, target_obj_id_mapping = (
                                scene.update_scene_graph(
                                    image_rgb=rgb[..., :3],
                                    depth=depth,
                                    intrinsics=cam_intr,
                                    cam_pos=cam_pose,
                                    pts=pts,
                                    pts_voxel=tsdf_planner.habitat2voxel(pts),
                                    img_path=obs_file_name,
                                    gt_bbox=metadata["gt_bbox"],
                                    gt_target_obj_ids=metadata["goal_obj_ids"],
                                )
                            )
                            scene.all_observations[obs_file_name] = rgb
                            rgb_egocentric_views.append(
                                resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                            )
                            if cfg.save_visualization:
                                plt.imsave(
                                    os.path.join(eps_snapshot_dir, obs_file_name),
                                    annotated_rgb,
                                    # resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                                )
                            else:
                                plt.imsave(
                                    os.path.join(eps_snapshot_dir, obs_file_name), rgb
                                )
                            # update the mapping of hm3d object id to our detected object id
                            for (
                                gt_goal_id,
                                det_goal_id,
                            ) in target_obj_id_mapping.items():
                                goal_obj_ids_mapping[gt_goal_id].append(det_goal_id)
                            pprint(goal_obj_ids_mapping)
                            all_added_obj_ids += added_obj_ids

                        # Clean up or merge redundant objects periodically
                        scene.periodic_cleanup_objects(
                            frame_idx=global_step * total_views + view_idx,
                            pts=pts,
                            goal_obj_ids_mapping=goal_obj_ids_mapping,
                        )
                        # Update depth map, occupancy map
                        tsdf_planner.integrate(
                            color_im=rgb,
                            depth_im=depth,
                            cam_intr=cam_intr,
                            cam_pose=pose_habitat_to_tsdf(cam_pose),
                            obs_weight=1.0,
                            margin_h=int(cfg.margin_h_ratio * img_height),
                            margin_w=int(cfg.margin_w_ratio * img_width),
                            explored_depth=cfg.explored_depth,
                        )

                    # (2) Update Memory Snapshots with hierarchical clustering
                    # Choose all the newly added objects as well as the objects nearby as the cluster targets
                    all_added_obj_ids = [
                        obj_id
                        for obj_id in all_added_obj_ids
                        if obj_id in scene.objects
                    ]
                    for obj_id, obj in scene.objects.items():
                        if (
                            np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]])
                            < cfg.scene_graph.obj_include_dist + 0.5
                        ):
                            all_added_obj_ids.append(obj_id)
                    scene.update_snapshots(
                        obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection
                    )
                    logging.info(
                        f"Step {global_step}, update snapshots, {len(scene.objects)} objects, {len(scene.snapshots)} snapshots"
                    )

                    # (3) Update the Frontier Snapshots
                    update_success = tsdf_planner.update_frontier_map(
                        pts=pts,
                        cfg=cfg.planner,
                        scene=scene,
                        cnt_step=global_step,
                        save_frontier_image=cfg.save_visualization,
                        eps_frontier_dir=eps_frontier_dir,
                        prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
                    )
                    if not update_success:
                        logging.info("Warning! Update frontier map failed!")

                    # (4) Choose the next navigation point by querying the VLM
                    if cfg.choose_every_step:
                        # if we choose to query vlm every step, we clear the target point every step
                        if (
                            tsdf_planner.max_point is not None
                            and type(tsdf_planner.max_point) == Frontier
                        ):
                            # reset target point to allow the model to choose again
                            tsdf_planner.max_point = None
                            tsdf_planner.target_point = None

                    # use the most common id in the mapped ids as the detected target object id
                    target_obj_ids_estimate = []
                    for obj_id, det_ids in goal_obj_ids_mapping.items():
                        if len(det_ids) == 0:
                            continue
                        target_obj_ids_estimate.append(
                            max(set(det_ids), key=det_ids.count)
                        )

                    if (
                        tsdf_planner.max_point is None
                        and tsdf_planner.target_point is None
                    ):
                        # query the VLM for the next navigation point, and the reason for the choice
                        try:
                            vlm_input_dict = prepare_vlm_input_dict(
                                metadata=metadata,
                                scene=scene,
                                tsdf_planner=tsdf_planner,
                                rgb_egocentric_views=rgb_egocentric_views,
                                tour_video=tour_video,
                                video_dir=video_dir,
                                cfg=cfg,
                                verbose=True,
                                global_step=global_step,
                            )
                            pred_out, max_point_choice = query_qwen(
                                vlm_input_dict,
                                cfg,
                                scene,
                                tsdf_planner,
                                model,
                                processor,
                            )
                        except Exception as error:
                            logging.exception(
                                "Failed to prepare object-selection candidates"
                            )
                            pred_out = {"error": f"{type(error).__name__}: {error}"}
                            max_point_choice = None
                        my_result[f"step_{global_step}"] = {
                            "step": global_step,
                            "agent_position": list(pts),
                            **pred_out,
                        }
                        if max_point_choice is None:
                            my_result["evaluation_status"] = "error"
                            my_result["failure_reason"] = pred_out["error"]
                            logging.error(
                                "Invalid object selection: %s", pred_out["error"]
                            )
                            break
                        step_log = {
                            "agent_position": list(pts),
                            "snapshots": list(scene.snapshots),
                            "frontiers": {},
                        }
                        # for imgid, snap in scene.snapshots.items():
                        #     step_log['snapshots'][imgid] = {
                        #         'position': snap.agent_position.tolist(),
                        #         'rotation': snap.agent_rotation,
                        #     }
                        for frontier in tsdf_planner.frontiers:
                            step_log["frontiers"][frontier.image] = {
                                "position": tsdf_planner.voxel2habitat(
                                    frontier.position.astype(int)
                                ).tolist(),
                                "orientation": frontier.orientation.tolist(),
                            }
                        my_result[f"step_{global_step}"].update(step_log)

                        # set the vlm choice as the navigation target
                        update_success = tsdf_planner.set_next_navigation_point(
                            choice=max_point_choice,
                            pts=pts,
                            objects=scene.objects,
                            cfg=cfg.planner,
                            pathfinder=scene.pathfinder,
                        )
                        if not update_success:
                            my_result["evaluation_status"] = "error"
                            my_result["failure_reason"] = (
                                "set_next_navigation_point failed"
                            )
                            logging.info(f"Invalid: set_next_navigation_point failed!")
                            break

                    # (5) Agent navigate to the target point for one step
                    return_values = tsdf_planner.agent_step_collect(
                        pts=pts,
                        angle=angle,
                        objects=scene.objects,
                        snapshots=scene.snapshots,
                        pathfinder=scene.pathfinder,
                        cfg=cfg.planner,
                        path_points=None,
                        save_visualization=cfg.save_visualization,
                    )
                    if return_values[0] is None:
                        my_result["evaluation_status"] = "error"
                        my_result["failure_reason"] = "agent_step_collect failed"
                        logging.info(f"Agent_step failed!")
                        break

                    # update agent's position and rotation
                    pts, angle, pts_voxel, fig, _, target_arrived = return_values
                    logger.log_step(pts_voxel=pts_voxel)
                    logging.info(f"Current position: {pts}, {logger.explore_dist:.3f}")

                    # sanity check about objects, scene graph, snapshots, ...
                    scene.sanity_check(cfg=cfg)

                    if cfg.save_visualization:
                        # save the top-down visualization
                        goal_position = episode["goals"][0]["position"]
                        goal_instance = episode["instruction"]["instance_id"][0]
                        goal_obs = list(goal_obj_ids_mapping.values())[0] != []
                        logger.save_topdown_visualization(
                            global_step=global_step,
                            goal_pos_voxel=tsdf_planner.habitat2voxel(goal_position),
                            goal_observed=goal_obs,
                            fig=fig,
                        )
                        # save the visualization of vlm's choice at each step
                        # logger.save_frontier_visualization(
                        #     global_step=global_step,
                        #     subtask_id='',
                        #     tsdf_planner=tsdf_planner,
                        #     max_point_choice=max_point_choice,
                        #     global_caption=f"{metadata['goal']}\n{metadata['class']}",
                        # )

                    # (6) Check if the agent has arrived at the target to finish the question
                    if type(max_point_choice) == SnapShot and target_arrived:
                        # when the target is a snapshot, and the agent arrives at the target
                        # we consider the subtask is finished, take an observation and save the chosen target snapshot
                        obs, _ = scene.get_observation(pts, angle=angle)
                        rgb = obs["color_sensor"]
                        plt.imsave(
                            os.path.join(logger.episode_dir, f"target.png"),
                            rgb,
                        )

                        # snapshot_filename = max_point_choice.image.split(".")[0]
                        #     f"cp {os.path.join(eps_snapshot_dir, max_point_choice.image)} {os.path.join(logger.subtask_object_observe_dir, f'snapshot_{snapshot_filename}.png')}"
                        # )

                        task_success = True
                        break

                # get some statistics
                if task_success and np.any(
                    [
                        obj_id in max_point_choice.cluster
                        for obj_id in target_obj_ids_estimate
                    ]
                ):
                    success_by_snapshot = True
                    logging.info(
                        f"Success: {target_obj_ids_estimate} in chosen snapshot {max_point_choice.image}!"
                    )
                else:
                    success_by_snapshot = False
                    logging.info(
                        f"Fail: {target_obj_ids_estimate} not in chosen snapshot!"
                    )
                # calculate the distance to the nearest view point
                agent_subtask_distance = calc_agent_subtask_distance(
                    pts, metadata["viewpoints"], scene.pathfinder
                )
                if agent_subtask_distance < cfg.success_distance:
                    success_by_distance = True
                    logging.info(
                        f"Success: agent reached the target viewpoint at distance {agent_subtask_distance}!"
                    )
                else:
                    success_by_distance = False
                    logging.info(
                        f"Fail: agent failed to reach the target viewpoint at distance {agent_subtask_distance}!"
                    )
                if success_by_distance:
                    break
                if attempt + 1 < max_attempts:
                    logging.info(
                        f"Retry episode {episode_id}: attempt {attempt + 1} failed at distance {agent_subtask_distance}"
                    )
            pl = metadata["gt_dist"] / max(metadata["gt_dist"], logger.explore_dist)
            my_result.update(
                {
                    "final_position": list(pts),
                    "success_by_snapshot": success_by_snapshot,
                    "success_by_distance": success_by_distance,
                    "dist_to_goal": agent_subtask_distance,
                    "nav_dist": logger.explore_dist,
                    "gt_dist": metadata["gt_dist"],
                    "spl_by_snapshot": success_by_snapshot * pl,
                    "spl_by_distance": success_by_distance * pl,
                }
            )
            logger.save_result(my_result)
            logging.info(f"Episode {episode_id} finish")

    if scene is not None:
        scene.close()


if __name__ == "__main__":
    run_cli(main, "config/eval_3dmem.yaml")
