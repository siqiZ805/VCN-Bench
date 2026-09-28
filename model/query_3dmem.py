import os
import re
import logging
from PIL import Image

from src.utils import resize_image

import torch
from transformers import AutoProcessor
from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from torchvision.transforms import v2


PROMPT = """First, select the best frame from the "Room Tour Video" as the global target; \
subsequently, choose either a "Snapshot" or a "Frontier" image in order to locate the target object specified in the "Navigation Goal"."""


def build_model(model_path, processor_path):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(processor_path)
    return model, processor


def prepare_vlm_input_dict(
    metadata, scene, tsdf_planner,
    rgb_egocentric_views: list, tour_video: list, video_dir,
    cfg, verbose=False, global_step=-1,
):
    step_dict = {}
    object_id_to_name = {
        obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
    }
    step_dict["obj_map"] = object_id_to_name
    step_dict["use_full_obj_list"] = cfg.use_full_obj_list
    step_dict["history_images"] = {}
    step_dict["current_images"] = {}
    step_dict["initial_view"] = {}

    rgb_ids = [x for x in list(scene.snapshots) if not x.startswith(f"{global_step}-")]
    for rgb_id in rgb_ids:
        step_dict["history_images"][rgb_id] = resize_image(
            scene.all_observations[rgb_id], cfg.prompt_h, cfg.prompt_w
        )

    cur_obs = [x for x in list(scene.snapshots) if x.startswith(f"{global_step}-")]
    for rgb_id in cur_obs:
        step_dict["current_images"][rgb_id] = resize_image(
            scene.all_observations[rgb_id], cfg.prompt_h, cfg.prompt_w
        )

    for view_idx in [3, 2, 0, 1]:
        rgb_id = f"0-view_{view_idx}.png"
        step_dict["initial_view"][rgb_id] = resize_image(
            scene.all_observations[rgb_id], cfg.prompt_h, cfg.prompt_w
        )

    step_dict["frontier_imgs"] = [
        frontier.feature for frontier in tsdf_planner.frontiers
    ]
    step_dict["egocentric_views"] = rgb_egocentric_views
    step_dict["use_egocentric_views"] = True
    step_dict["goal"] = metadata["goal"]
    step_dict["class"] = metadata["class"]
    step_dict["tour_video"] = tour_video
    step_dict["video_dir"] = video_dir
    step_dict["video_root"] = cfg.video_root
    return step_dict


def format_item(input_dict):
    source = [
        {"from": "human", "value": ""},
        {"from": "gpt", "value": ""}
    ]

    video_list = [f"{i} <image>" for i in range(len(input_dict["tour_video"]))]
    source[0]["value"] = f"- Room Tour Video: {', '.join(video_list)}\n"

    hist_count = len(input_dict["history_images"])
    if hist_count > 0:
        source[0]["value"] += f"- History: {'<image>' * hist_count}"

    directions = ["front", "left", "back", "right"]
    init_list = [f"{d} <image>" for d in directions]
    source[0]["value"] += f"- Initial View: {' '.join(init_list)}\n"

    cur_count = len(input_dict["current_images"])
    if cur_count > 0:
        cur_list = [f"{i} <image>" for i in range(cur_count)]
        source[0]["value"] += f"- Snapshot: {', '.join(cur_list)}\n"
    else:
        source[0]["value"] += "- Snapshot: No snapshots.\n"

    frontier_count = len(input_dict["frontier_imgs"])
    if frontier_count > 0:
        frontier_list = [f"{i} <image>" for i in range(frontier_count)]
        source[0]["value"] += f"- Frontier: {', '.join(frontier_list)}\n"
    else:
        source[0]["value"] += "- Frontier: No frontiers.\n"

    goal = input_dict["goal"]
    if ".png" in goal:
        source[0]["value"] += "- Navigation Goal: <image>\n"
    else:
        source[0]["value"] += f"- Navigation Goal: {goal}\n"
    source[0]["value"] += "\n" + PROMPT

    transform = v2.Resize([384, 384])
    image_pool = []
    for img in input_dict["tour_video"]:
        image_pool.append({
            "type": "image",
            "image": transform(Image.open(os.path.join(input_dict["video_dir"], img)).convert("RGB"))
        })

    snap_imgs = [img for _, img in input_dict["history_images"].items()] \
        + [img for _, img in input_dict["initial_view"].items()] \
        + [img for _, img in input_dict["current_images"].items()] \
        + input_dict["frontier_imgs"]
    for img in snap_imgs:
        image_pool.append({
            "type": "image",
            "image": transform(Image.fromarray(img).convert("RGB"))
        })

    if ".png" in goal:
        image_pool.append({
            "type": "image",
            "image": transform(Image.open(os.path.join(input_dict["video_root"], goal)).convert("RGB"))
        })

    messages = []
    for turn in source:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]
        if role == "user":
            content = []
            text_parts = re.split(r"(<image>|<video>)", text)
            for seg in text_parts:
                if seg == "<image>":
                    content.append(image_pool.pop(0))
                else:
                    content.append({"type": "text", "text": seg.strip()})
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    return source, messages


def replace_image_placeholders_batch(
    input_ids, high_res_len,
    anchor_token_id=151652, placeholder_token_id=151655, target_num=9, pad_token_id=151643,
):
    """Trim image placeholders per sample, then pad the batch to a common length."""
    device = input_ids.device
    processed_seqs = []
    for b_idx in range(input_ids.shape[0]):
        single_input_ids = input_ids[b_idx]
        anchor_positions = (single_input_ids == anchor_token_id).nonzero(as_tuple=True)[0]
        anchor_positions = anchor_positions[:-high_res_len]
        new_seq = single_input_ids.clone()
        for anchor_idx in reversed(anchor_positions):
            anchor_idx = anchor_idx.item()
            placeholder_start = anchor_idx + 1
            placeholder_end = placeholder_start
            while placeholder_end < new_seq.shape[0] and new_seq[placeholder_end] == placeholder_token_id:
                placeholder_end += 1
            prefix = new_seq[:anchor_idx + 1]
            new_placeholders = torch.tensor(
                [placeholder_token_id] * target_num, dtype=torch.long, device=device
            )
            suffix = new_seq[placeholder_end:]
            new_seq = torch.cat([prefix, new_placeholders, suffix], dim=0)
        processed_seqs.append(new_seq)

    return torch.nn.utils.rnn.pad_sequence(
        sequences=processed_seqs,
        batch_first=True,
        padding_value=pad_token_id,
    )


def query_qwen(input_dict, cfg, scene, tsdf_planner, model, processor):
    _, messages = format_item(input_dict)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )

    high_res_len = 4 + len(input_dict["current_images"]) + len(input_dict["frontier_imgs"])
    if ".png" in input_dict["goal"]:
        high_res_len += 1
    img_num = inputs["image_grid_thw"].size(0)
    high_res_mask = torch.zeros(img_num, dtype=torch.bool)
    high_res_mask[-high_res_len:] = True
    inputs["high_res_mask"] = high_res_mask
    inputs["input_ids"] = replace_image_placeholders_batch(inputs["input_ids"], high_res_len)

    low_res_len = img_num - high_res_len
    my_grid_thw = []
    for i in range(img_num):
        if i < low_res_len:
            my_grid_thw.append([1, 6, 6])
        else:
            my_grid_thw.append([1, 24, 24])
    inputs["my_grid_thw"] = torch.tensor(my_grid_thw, dtype=inputs["image_grid_thw"].dtype)
    inputs["attention_mask"] = inputs["input_ids"].ne(processor.tokenizer.pad_token_id)

    with torch.no_grad():
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logging.info(output_text)

    try:
        out1, out2 = output_text[0].split(". Answer:")
        out_frame = int(out1.split("Frame:")[-1].strip())
        out_answer = out2.split(".")[0].strip()
        target_type, target_index = out_answer.split(" ")
        target_index = int(target_index.strip())
        assert target_type in ["snapshot", "frontier"], f"Wrong target type: {target_type}, failed!"
    except Exception:
        return -1, None

    if target_type == "snapshot":
        if target_index < 0 or target_index >= len(input_dict["current_images"]):
            logging.info(f"Target index can not match real objects: {target_index}, failed!")
            return out_frame, None
        rgb_name = list(input_dict["current_images"])[target_index]
        if rgb_name not in scene.snapshots:
            logging.info(f"{rgb_name} not in snapshots: {list(scene.snapshots)}. Failed")
            return out_frame, None
        max_point_choice = scene.snapshots[rgb_name]
    else:
        if target_index < 0 or target_index >= len(tsdf_planner.frontiers):
            logging.info(f"Predicted frontier target index out of range: {target_index}, failed!")
            return out_frame, None
        max_point_choice = tsdf_planner.frontiers[target_index]

    return (out_frame, target_type, target_index), max_point_choice
