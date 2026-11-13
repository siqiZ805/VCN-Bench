"""Generate and validate instruction datasets using a portable YAML configuration."""

import argparse
import gzip
import json
import os
import random
from collections import Counter

import numpy as np
from tqdm import tqdm

from src.io_utils import load_json
from src.runtime import configure_environment, load_config

from .config import load_construction_config, worker_shard
from .constants import TASK_TYPE_TO_INDEX, TEST_SCENES
from .utils import abstract_traj

configure_environment()


def construct_sim(scan, config):
    from ..habitat_utils.sim import HabitatSimulator

    sim_settings = load_config(config.sim_config_path)
    agent_setting = load_config(config.agent_config_path)
    sim_settings["scene"] = os.path.join(config.mp3d_dir, scan, scan + ".glb")
    abstract_sim = HabitatSimulator(sim_settings, agent_setting)
    sim = abstract_sim.simulator
    agent = abstract_sim.agent
    pathfinder = sim.pathfinder
    return sim, agent, pathfinder


def main(config, task_type, func, mode, rank=0, world_size=1):
    results = {
        "episodes": [],
    }
    sim = None
    scans = sorted(
        name
        for name in os.listdir(config.tour_video_root)
        if os.path.isdir(os.path.join(config.tour_video_root, name))
    )
    # scans = ['8194nk5LbLH', 'EU6Fwq7SyZv', 'zsNo4HB9uLZ', 'oLBMNvg9in8', 'pLe4wQe7qrG', 'Z6MFQCViBuw']
    scene_region_dists = load_json(config.region_dist_path)
    all_regions = load_json(config.regions_path)

    if task_type in ["obj_instance", "obj_category"]:
        det_step = 50
        window_step = 100
    else:
        det_step = 1
        window_step = 10

    for scan in tqdm(scans[rank::world_size]):
        if mode == "train" and scan in TEST_SCENES:
            continue
        if task_type == "obj_category":
            if os.path.exists(
                os.path.join(config.output_dir, f"{task_type}_{scan}.json")
            ):
                continue
            if sim is not None:
                sim.close()
            sim, agent, pathfinder = construct_sim(scan, config)
            results = {
                "episodes": [],
            }

        objects_dict = load_json(
            os.path.join(config.scene_info_path, scan, "object_dict.json")
        )
        inst_viewpoints = load_json(os.path.join(config.viewpoints_dir, scan + ".json"))
        inst_instructions = load_json(
            os.path.join(
                config.scene_info_path, scan, "object_instruction_with_room.json"
            )
        )

        scan_dir = os.path.join(config.tour_video_root, scan)
        heights = sorted(
            x for x in os.listdir(scan_dir) if os.path.isdir(os.path.join(scan_dir, x))
        )

        for hkey in heights:
            ori_trajectory = load_json(f"{config.trajectory_dir}/{scan}_{hkey}.json")[
                "trajectory"
            ]
            ori_rooms = load_json(f"{config.trajectory_dir}/{scan}_{hkey}_rooms.json")
            ori_detailed_rooms = ori_rooms["detail"]

            ds_trajectory = load_json(f"{scan_dir}/prior_{hkey}.json")["trajectory"]
            ds_rooms = load_json(f"{scan_dir}/rooms_{hkey}.json")
            ds_detailed_rooms = ds_rooms["detail"]
            ds_abs_rooms = ds_rooms["abstract"]
            ds_insts = load_json(f"{scan_dir}/instances_{hkey}.json")

            all_episodes = []
            det_point = 0
            for abs_idx, start_room in enumerate(ds_abs_rooms):
                while (
                    det_point < len(ds_detailed_rooms)
                    and ds_detailed_rooms[det_point] != start_room
                ):
                    det_point += det_step
                if det_point >= len(ds_detailed_rooms) - 10:
                    break

                max_traj_len = len(ds_detailed_rooms) - det_point
                cand_lens = np.arange(10, max_traj_len, window_step)

                for traj_len in cand_lens:
                    end_idx = det_point + traj_len
                    if end_idx >= len(ds_detailed_rooms):
                        continue
                    input_dict = {
                        "ori_trajectory": ori_trajectory,
                        "ori_detailed_rooms": ori_detailed_rooms,
                        "trajectory": ds_trajectory[det_point:end_idx],
                        "detailed_rooms": ds_detailed_rooms[det_point:end_idx],
                        "traj_insts": ds_insts[det_point:end_idx],
                    }
                    input_dict["abs_rooms"] = abstract_traj(
                        input_dict["detailed_rooms"]
                    )
                    room_set = set([tuple(x) for x in input_dict["abs_rooms"]])
                    if len(list(room_set)) < 2:
                        continue
                    input_dict.update(
                        {
                            "room_set": list(room_set),
                            "objects_dict": objects_dict,
                            "inst_viewpoints": inst_viewpoints,
                            "inst_instructions": inst_instructions,
                            "scene_region_dists": scene_region_dists[scan],
                            "all_regions": all_regions[scan],
                            # 'scene_NSWE': scene_NSWE[scan],
                            "scan": scan,
                        }
                    )
                    # -----------------------------
                    if task_type == "obj_category":
                        episodes = func(input_dict, pathfinder)
                    else:
                        episodes = func(input_dict, mode)

                    for ep in episodes:
                        ep.update(
                            {
                                "scene_id": f"mp3d/{scan}/{scan}.glb",
                                "hkey": hkey,
                                "start_idx": det_point,
                                "end_idx": int(end_idx),
                            }
                        )
                        all_episodes.append(ep)

            all_tags = set()
            random.shuffle(all_episodes)
            for ep in all_episodes:
                tag = ep["tag"]
                if tag in all_tags:
                    continue
                all_tags.add(tag)
                results["episodes"].append(ep)

        if task_type == "obj_category":
            with open(
                os.path.join(config.output_dir, f"{task_type}_{scan}.json"), "w"
            ) as f:
                print(task_type, scan, len(results["episodes"]))
                json.dump(results, f)

    if task_type != "obj_category":
        with open(
            os.path.join(
                config.output_dir,
                (
                    f"{task_type}_rank{rank}.json"
                    if world_size > 1
                    else task_type + ".json"
                ),
            ),
            "w",
        ) as f:
            print(task_type, len(results["episodes"]))
            json.dump(results, f)

    if sim is not None:
        sim.close()


def validate_navigation_episodes(config, all_episodes):
    """Retain reachable goals at least 3m away and assign task-specific IDs."""
    import habitat_sim

    task_count = dict.fromkeys(TASK_TYPE_TO_INDEX, 0)
    new_episodes = []
    sim = None
    scans = sorted(
        name
        for name in os.listdir(config.tour_video_root)
        if os.path.isdir(os.path.join(config.tour_video_root, name))
    )
    for scan in scans:
        heights = [
            x
            for x in os.listdir(os.path.join(config.tour_video_root, scan))
            if ".json" not in x and "_" not in x
        ]
        if sim is not None:
            sim.close()
        sim, agent, pathfinder = construct_sim(scan, config)

        for hkey in heights:
            prior = load_json(
                os.path.join(config.tour_video_root, scan, f"prior_{hkey}.json")
            )
            scanh_episodes = [
                x for x in all_episodes if scan in x["scene_id"] and x["hkey"] == hkey
            ]
            if len(scanh_episodes) == 0:
                continue

            start_state = prior["trajectory"][0]
            agent_state = habitat_sim.AgentState()
            agent_state.position = start_state["position"]
            agent_state.rotation = start_state["rotation"]
            agent.set_state(agent_state)

            for ep in tqdm(scanh_episodes, desc=f"{scan} {hkey}"):
                tt = ep["instruction"]["task_type"]
                task_count[tt] += 1

                path = habitat_sim.MultiGoalShortestPath()
                path.requested_start = ep["start_position"]
                path.requested_ends = [
                    x["agent_state"]["position"]
                    for goal in ep["goals"]
                    for x in goal["view_points"]
                ]
                pathfinder.find_path(path)
                dist = path.geodesic_distance
                if dist < 3 or dist == np.inf:
                    continue
                ep["dist_to_goal"] = dist
                ep["episode_id"] = f"{TASK_TYPE_TO_INDEX[tt]}_{task_count[tt]}"
                new_episodes.append(ep)

    if sim is not None:
        sim.close()
    return new_episodes


def merge_and_validate(config):
    files = [
        x
        for x in os.listdir(config.output_dir)
        if x.endswith(".json") and "rephrase" not in x
    ]
    all_episodes = []
    for file in files:
        data = load_json(os.path.join(config.output_dir, file))["episodes"]
        if "category" not in file:
            print(file, len(data))
            random.shuffle(data)
            all_episodes += data[:500]
        else:
            all_episodes += data

    # task_types = set([x['instruction']['task_type'] for x in all_episodes])
    # task_to_idx = {tt:idx for idx,tt in enumerate(list(task_types))}
    print(TASK_TYPE_TO_INDEX)
    high_task_types = sorted({x.split("/")[0] for x in TASK_TYPE_TO_INDEX})

    new_episodes = validate_navigation_episodes(config, all_episodes)

    select_episodes = []
    for high_tt in high_task_types:
        if high_tt == "room_temporal_count":
            cur_room_eps = [
                x
                for x in new_episodes
                if x["instruction"]["task_type"] == "room_temporal_count/cur_room"
            ]
            other_room_eps = [
                x
                for x in new_episodes
                if x["instruction"]["task_type"] == "room_temporal_count/other_room"
            ]
            random.shuffle(other_room_eps)
            select_episodes += cur_room_eps
            select_episodes += other_room_eps[: 250 - len(cur_room_eps)]
        else:
            tt_eps = [
                x for x in new_episodes if high_tt in x["instruction"]["task_type"]
            ]
            random.shuffle(tt_eps)
            select_episodes += tt_eps[:250]

    results = {
        "task_to_idx": TASK_TYPE_TO_INDEX,
        "episodes": select_episodes,
        "category_to_task_category_id": {},
        "category_to_scene_annotation_category_id": {},
    }
    epids = list(set([x["episode_id"] for x in select_episodes]))
    print(len(select_episodes), len(epids))
    assert len(select_episodes) == len(epids)
    final_task_count = Counter([x["instruction"]["task_type"] for x in select_episodes])
    print(final_task_count)
    with gzip.open(f"{config.output_dir}/all.json.gz", "w") as f:
        f.write(json.dumps(results).encode("utf-8"))


def merge_and_validate_train(config, do_cat=False):
    files = [
        x
        for x in os.listdir(config.output_dir)
        if x.endswith(".json") and "rephrase" not in x
    ]
    all_episodes = []
    # cat_episodes = []
    for file in files:
        if do_cat == False:
            if "category" not in file:
                data = load_json(os.path.join(config.output_dir, file))["episodes"]
                print(file, len(data))
                all_episodes += data[:10000]
        else:
            if "category" in file:
                data = load_json(os.path.join(config.output_dir, file))["episodes"]
                print(file, len(data))
                all_episodes += data
    if do_cat:
        print("obj_category:", len(all_episodes))

    print(TASK_TYPE_TO_INDEX)
    new_episodes = validate_navigation_episodes(config, all_episodes)

    results = {
        "task_to_idx": TASK_TYPE_TO_INDEX,
        "episodes": new_episodes,
        "category_to_task_category_id": {},
        "category_to_scene_annotation_category_id": {},
    }
    epids = list(set([x["episode_id"] for x in new_episodes]))
    print(len(new_episodes), len(epids))
    assert len(new_episodes) == len(epids)
    final_task_count = Counter([x["instruction"]["task_type"] for x in new_episodes])
    print(final_task_count)

    if do_cat:
        output_path = f"{config.output_dir}/instr_cat.json.gz"
    else:
        output_path = f"{config.output_dir}/instr_nocat.json.gz"
    with gzip.open(output_path, "w") as f:
        f.write(json.dumps(results).encode("utf-8"))


TASKS = (
    "spatial_distance",
    "spatial_orientation",
    "temporal_count",
    "temporal_order_multi",
    "temporal_order_sing",
    "obj_instance",
    "obj_category",
)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_file", default="config/construct_instructions.yaml")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--action", choices=("generate", "merge"), default="generate")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument(
        "--category_only", action="store_true", help="Merge category training episodes"
    )
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world_size", type=int)
    args = parser.parse_args()
    if args.action == "generate" and args.task is None:
        parser.error("--task is required for generation")
    rank, world_size = worker_shard(args.rank, args.world_size)
    if args.action == "merge" and world_size != 1:
        parser.error("Merge with one worker after generation has finished")
    config = load_construction_config(args.cfg_file, args.mode)
    os.makedirs(config.output_dir, exist_ok=True)
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    if args.action == "merge":
        if args.mode == "test":
            merge_and_validate(config)
        else:
            merge_and_validate_train(config, do_cat=args.category_only)
        return
    from .object import obj_category, obj_instance
    from .spatial import spatial_distance, spatial_orientation_rotation
    from .temporal import temporal_count, temporal_order_multi, temporal_order_sing

    functions = dict(
        zip(
            TASKS,
            (
                spatial_distance,
                spatial_orientation_rotation,
                temporal_count,
                temporal_order_multi,
                temporal_order_sing,
                obj_instance,
                obj_category,
            ),
        )
    )
    main(
        config,
        args.task,
        functions[args.task],
        args.mode,
        rank=rank,
        world_size=world_size,
    )


if __name__ == "__main__":
    cli()
