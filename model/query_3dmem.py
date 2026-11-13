"""Zero-shot Qwen3-VL adapter for the official 3D-Mem object-selection interface."""

import copy
import hashlib
import json
import logging
import os
import re
from dataclasses import replace

import numpy as np
from PIL import Image

from src.tour_video import SAMPLING_PROTOCOL

EVALUATION_PROTOCOL = "3dmem_qwen3vl_zero_shot_object_v1"
PROMPT = """You are an indoor navigation agent.

Your task is to identify the destination specified by the Navigation Goal using
the prior Room Tour Video, then choose your next navigation target from the
available Memory Snapshots or Frontiers.

Inputs:
- Room Tour Video: sampled frames in chronological order. Frame indices refer
  to the displayed sequence.
- Initial Views: observations at the episode's starting location.
- Current Views: observations at the agent's current location.
- Memory Snapshots: observations collected during navigation, including earlier
  steps. Each snapshot contains a full image followed by numbered object crops
  and detected class names.
- Frontiers: unexplored regions that the agent can visit.

Instructions:
1. Use the Room Tour Video and Navigation Goal to resolve the intended
   destination, including any spatial, temporal, or instance relations. Select
   the video frame that best shows the intended destination.
2. Compare the destination with the available Memory Snapshots. If you can
   identify the intended target object in a snapshot, select BOTH that snapshot
   and the corresponding object crop.
3. Selecting a snapshot commits to approaching the selected object and ending
   navigation upon arrival. A matching object category alone is insufficient
   when the goal specifies a particular instance.
4. If the target cannot be identified among the snapshot objects, select a
   Frontier that is most likely to help locate it.
5. Detected class names may be inaccurate. Use the full images, object crops,
   and video context to make your decision.
6. Use only the displayed indices. Object indices are local to each snapshot.
   Video frames and context views are not navigation candidates.

Return exactly one line in one of these formats:
Frame: k. Answer: Snapshot i, Object j
Frame: k. Answer: Frontier i

All indices are zero-based. Do not add explanations."""


def evaluation_identity(config):
    """Bind result reuse to this protocol and the resolved evaluation config."""
    settings = {
        k: v
        for k, v in config.items()
        if k not in {"output_dir", "output_parent_dir", "exp_name"}
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "evaluation_signature": digest,
        "tour_video_sampling": SAMPLING_PROTOCOL,
    }


def result_is_reusable(result, identity):
    if any(result.get(key) != value for key, value in identity.items()):
        raise ValueError(
            "Existing result uses a different evaluation protocol/config; "
            "choose a new exp_name or output directory."
        )
    return result.get("evaluation_status") == "completed"


def build_model(model_path, processor_path=None):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()
    processor = AutoProcessor.from_pretrained(processor_path or model_path)
    return model, processor


def _rgb(image, size=None):
    image = (
        image.convert("RGB")
        if isinstance(image, Image.Image)
        else Image.fromarray(image).convert("RGB")
    )
    return image.resize(size) if size is not None else image


def _load_image(path, size):
    with Image.open(path) as image:
        return _rgb(image, size)


def prepare_vlm_input_dict(
    metadata,
    scene,
    tsdf_planner,
    rgb_egocentric_views,
    tour_video,
    video_dir,
    cfg,
    verbose=False,
    global_step=-1,
):
    size = (cfg.prompt_w, cfg.prompt_h)
    snapshots = []
    for image_name, snapshot in scene.snapshots.items():
        visual_prompt = getattr(snapshot, "visual_prompt", None)
        if image_name not in scene.all_observations or visual_prompt is None:
            logging.warning("Skipping snapshot without RGB/detections: %s", image_name)
            continue
        boxes = np.asarray(visual_prompt.xyxy)
        obj_ids = np.asarray(visual_prompt.data.get("obj_id", [])).reshape(-1)
        if len(boxes) != len(obj_ids):
            raise ValueError(f"Detection/object ID count mismatch in {image_name}")
        allowed = set(
            snapshot.full_obj_list if cfg.use_full_obj_list else snapshot.cluster
        )
        rgb = _rgb(scene.all_observations[image_name])
        objects, seen = [], set()
        for box, object_id in zip(boxes, obj_ids):
            object_id = int(object_id)
            if (
                object_id not in allowed
                or object_id not in scene.objects
                or object_id in seen
            ):
                continue
            box = np.asarray(box, dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            x1, y1 = np.floor(box[:2]).astype(int)
            x2, y2 = np.ceil(box[2:]).astype(int)
            x1, x2 = np.clip([x1, x2], 0, rgb.width).tolist()
            y1, y2 = np.clip([y1, y2], 0, rgb.height).tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            crop = rgb.crop((x1, y1, x2, y2))
            crop.thumbnail(
                size
            )  # Preserve aspect ratio; the standard processor encodes it.
            objects.append(
                {
                    "obj_id": object_id,
                    "obj_class": scene.objects[object_id]["class_name"],
                    "bbox": [x1, y1, x2, y2],
                    "crop": crop,
                }
            )
            seen.add(object_id)
        if objects:
            snapshots.append(
                {
                    "image": image_name,
                    "full_img": rgb.resize(size),
                    "cluster": list(snapshot.cluster),
                    "objects": objects,
                }
            )

    initial = [
        (name, _rgb(scene.all_observations[name], size))
        for name in [f"0-view_{idx}.png" for idx in [3, 2, 0, 1]]
        if name in scene.all_observations
    ]
    return {
        "snapshots": snapshots,
        "obj_map": {
            int(key): value["class_name"] for key, value in scene.objects.items()
        },
        "initial_views": initial,
        "current_views": (
            [_rgb(image, size) for image in rgb_egocentric_views]
            if cfg.egocentric_views
            else []
        ),
        "frontier_imgs": [
            _rgb(frontier.feature, size) for frontier in tsdf_planner.frontiers
        ],
        "frontier_names": [frontier.image for frontier in tsdf_planner.frontiers],
        "goal": metadata["goal"],
        "tour_video": list(tour_video),
        "video_dir": video_dir,
        "video_root": cfg.video_root,
        "image_size": size,
    }


def _text(content, text):
    content.append({"type": "text", "text": text})


def _image(content, label, image):
    _text(content, label)
    content.append({"type": "image", "image": image})


def _goal(content, input_dict):
    goal = input_dict["goal"]
    if str(goal).lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        _image(
            content,
            "Navigation Goal (target image):",
            _load_image(
                os.path.join(input_dict["video_root"], goal), input_dict["image_size"]
            ),
        )
    else:
        _text(content, f"Navigation Goal: {goal}")


def format_item(input_dict):
    content = []
    _goal(content, input_dict)
    _text(content, "Room Tour Video (chronological sampled frames):")
    for index, name in enumerate(input_dict["tour_video"]):
        _image(
            content,
            f"Frame {index}:",
            _load_image(
                os.path.join(input_dict["video_dir"], name), input_dict["image_size"]
            ),
        )
    _text(content, "Initial Views (context only):")
    initial_directions = {
        "0-view_3.png": "front",
        "0-view_2.png": "left",
        "0-view_0.png": "back",
        "0-view_1.png": "right",
    }
    for index, (name, image) in enumerate(input_dict["initial_views"]):
        _image(
            content,
            f"Initial View {index} ({initial_directions[name]} at episode start):",
            image,
        )
    _text(content, "Current Views (context only):")
    for index, image in enumerate(input_dict["current_views"]):
        _image(content, f"Current View {index}:", image)
    _text(
        content,
        (
            "Available Memory Snapshots:"
            if input_dict["snapshots"]
            else "No Snapshot is available."
        ),
    )
    for index, snapshot in enumerate(input_dict["snapshots"]):
        _image(content, f"Snapshot {index} (full image):", snapshot["full_img"])
        for object_index, obj in enumerate(snapshot["objects"]):
            _image(content, f"Object {object_index}: {obj['obj_class']}", obj["crop"])
    _text(
        content,
        (
            "Available Frontiers:"
            if input_dict["frontier_imgs"]
            else "No Frontier is available."
        ),
    )
    for index, image in enumerate(input_dict["frontier_imgs"]):
        _image(content, f"Frontier {index}:", image)
    _text(content, "Return your selection using the exact output format above.")
    return [
        {"role": "system", "content": [{"type": "text", "text": PROMPT}]},
        {"role": "user", "content": content},
    ]


def _generate(messages, model, processor, max_new_tokens):
    import torch

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
    )
    text_config = getattr(model.config, "text_config", model.config)
    context_limit = getattr(text_config, "max_position_embeddings", None)
    input_tokens = inputs["input_ids"].shape[-1]
    if context_limit and input_tokens + max_new_tokens > context_limit:
        raise ValueError(
            f"Input has {input_tokens} tokens; generation would exceed "
            f"the model context limit {context_limit}. No inputs were truncated."
        )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return processor.batch_decode(
        generated[:, input_tokens:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _retry(messages, response, error):
    return messages + [
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Invalid response: {error}. Use only the displayed indices and the exact "
                    "requested format. Return the corrected answer without explanations.",
                }
            ],
        },
    ]


def _prefilter(input_dict, cfg, model, processor, record):
    classes = sorted(
        {
            obj["obj_class"]
            for snap in input_dict["snapshots"]
            for obj in snap["objects"]
        }
    )
    if not classes or not cfg.prefiltering:
        return input_dict
    if cfg.top_k_categories < 1:
        raise ValueError("top_k_categories must be positive")
    content = []
    _goal(content, input_dict)
    _text(
        content,
        "Rank detected object categories that may identify the destination or help "
        "locate it. Return only exact category names from the list, one per line, at most "
        f"{cfg.top_k_categories} categories. Return NONE if no category is relevant. "
        "Do not infer a particular instance from its category alone.\nCategories:\n"
        + "\n".join(classes),
    )
    messages = [{"role": "user", "content": content}]
    for _ in range(3):
        response = _generate(messages, model, processor, 256)
        selected = (
            []
            if response == "NONE"
            else [line.strip() for line in response.splitlines()]
        )
        error = None
        if response != "NONE" and (
            not selected
            or any(name not in classes for name in selected)
            or len(set(selected)) != len(selected)
            or len(selected) > cfg.top_k_categories
        ):
            error = "Return at most the requested number of exact, unique category names, or NONE"
        record["prefilter_responses"].append({"response": response, "error": error})
        if error is None:
            break
        messages = _retry(messages, response, error)
    else:
        raise ValueError("Category prefiltering failed after three attempts")
    record["selected_categories"] = selected
    selected = set(selected)
    snapshots = []
    for snapshot in input_dict["snapshots"]:
        # Official filtering retains snapshots whose cluster has a relevant class.
        if not any(
            input_dict["obj_map"].get(int(obj_id)) in selected
            for obj_id in snapshot["cluster"]
        ):
            continue
        objects = [obj for obj in snapshot["objects"] if obj["obj_class"] in selected]
        if objects:
            snapshots.append({**snapshot, "objects": objects})
    return {**input_dict, "snapshots": snapshots}


def candidate_mapping(input_dict):
    return {
        "video_frames": list(input_dict["tour_video"]),
        "snapshots": [
            {
                "snapshot_index": index,
                "image": snapshot["image"],
                "objects": [
                    {
                        "object_index": obj_index,
                        **{key: obj[key] for key in ("obj_id", "obj_class", "bbox")},
                    }
                    for obj_index, obj in enumerate(snapshot["objects"])
                ],
            }
            for index, snapshot in enumerate(input_dict["snapshots"])
        ],
        "frontiers": [
            {"frontier_index": index, "image": name}
            for index, name in enumerate(input_dict["frontier_names"])
        ],
    }


def parse_selection(response, input_dict):
    match = re.fullmatch(
        r"Frame:\s*(\d+)\.\s*Answer:\s*(?:Snapshot\s+(\d+),\s*Object\s+(\d+)|Frontier\s+(\d+))",
        response.strip(),
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            "Expected 'Frame: k. Answer: Snapshot i, Object j' or 'Frame: k. Answer: Frontier i'"
        )
    frame = int(match[1])
    if frame >= len(input_dict["tour_video"]):
        raise ValueError("Frame index out of range")
    result = {
        "pred_frame_idx": frame,
        "pred_obj_idx": None,
        "pred_obj_id": None,
        "pred_obj_class": None,
        "pred_snapshot_image": None,
    }
    if match[2] is not None:
        index, obj_index = int(match[2]), int(match[3])
        if index >= len(input_dict["snapshots"]):
            raise ValueError("Snapshot index out of range")
        snapshot = input_dict["snapshots"][index]
        if obj_index >= len(snapshot["objects"]):
            raise ValueError("Object index out of range for the selected snapshot")
        obj = snapshot["objects"][obj_index]
        result.update(
            pred_snap_type="snapshot",
            pred_snap_idx=index,
            pred_obj_idx=obj_index,
            pred_obj_id=obj["obj_id"],
            pred_obj_class=obj["obj_class"],
            pred_snapshot_image=snapshot["image"],
        )
    else:
        index = int(match[4])
        if index >= len(input_dict["frontier_imgs"]):
            raise ValueError("Frontier index out of range")
        result.update(pred_snap_type="frontier", pred_snap_idx=index)
    return result


def navigation_choice(prediction, scene, tsdf_planner):
    if prediction["pred_snap_type"] == "frontier":
        return tsdf_planner.frontiers[prediction["pred_snap_idx"]]
    object_id = prediction["pred_obj_id"]
    if object_id not in scene.objects:
        raise ValueError("Selected object no longer exists in the scene map")
    snapshot = scene.snapshots[prediction["pred_snapshot_image"]]
    return replace(
        snapshot,
        obs_point=copy.deepcopy(snapshot.obs_point),
        full_obj_list={object_id: scene.objects[object_id]["conf"]},
        cluster=[object_id],
    )


def query_qwen(input_dict, cfg, scene, tsdf_planner, model, processor):
    """Return a JSON-safe prediction/audit record and a singleton snapshot or frontier."""
    record = {
        "raw_response": None,
        "responses": [],
        "prefilter_responses": [],
        "selected_categories": None,
        "error": None,
        "candidate_mapping": candidate_mapping(input_dict),
    }
    try:
        if not input_dict["tour_video"]:
            raise ValueError("Room Tour Video has no sampled frames")
        input_dict = _prefilter(input_dict, cfg, model, processor, record)
        record["candidate_mapping"] = candidate_mapping(input_dict)
        if not input_dict["snapshots"] and not input_dict["frontier_imgs"]:
            raise ValueError("No valid snapshot objects or frontiers are available")
        messages = format_item(input_dict)
        for _ in range(3):
            response = _generate(messages, model, processor, 64)
            record["raw_response"] = response
            try:
                prediction = parse_selection(response, input_dict)
                choice = navigation_choice(prediction, scene, tsdf_planner)
            except (ValueError, KeyError, IndexError) as error:
                record["responses"].append({"response": response, "error": str(error)})
                messages = _retry(messages, response, str(error))
                continue
            record["responses"].append({"response": response, "error": None})
            record.update(prediction)
            logging.info("Object-level navigation selection: %s", prediction)
            return record, choice
        raise ValueError("Navigation selection failed after three attempts")
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        logging.exception("Zero-shot 3D-Mem query failed")
        return record, None
