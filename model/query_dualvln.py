import logging
import os
import re
from collections import deque

from PIL import Image

PROMPT = """First, select the best frame from the "Room Tour Video" as the global target; \
subsequently, choose a view from "Current Observations" and generate a pixel goal."""


def build_model(model_path, processor_path):
    import torch
    from transformers import AutoProcessor

    from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(processor_path or model_path)
    return model, processor


def format_item(
    tour_video, hist_rgbs, init_rgbs, ego_rgbs, video_dir, video_root, nav_goal
):
    source = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]

    video_list = [f"{i} <image>" for i in range(len(tour_video))]
    source[0]["value"] = f"- Room Tour Video: {', '.join(video_list)}\n"

    hist_count = len(hist_rgbs)
    if hist_count > 0:
        # hist_list = [f"{i} <image>" for i in range(hist_count)]
        # source[0]['value'] += f"- Navigation History: {', '.join(hist_list)}"
        source[0]["value"] += "- Navigation History: " + "<image>" * hist_count + "\n"
    else:
        source[0]["value"] += "- Navigation History: None.\n"

    directions = ["front", "left", "back", "right"]
    init_obs_list = [f"{d}<image>" for d in directions]
    source[0]["value"] += f"- Initial Observations: {' '.join(init_obs_list)}\n"
    cur_obs_list = [f"{d} <image>" for d in directions]
    source[0]["value"] += f"- Current Observations: {' '.join(cur_obs_list)}\n"

    image_goal = str(nav_goal).lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    source[0]["value"] += (
        "- Navigation Goal: <image>\n"
        if image_goal
        else f"- Navigation Goal: {nav_goal}\n"
    )

    source[0]["value"] += "\n" + PROMPT

    image_pool = []
    for img in tour_video:
        with Image.open(os.path.join(video_dir, img)) as image:
            image_pool.append(
                {
                    "type": "image",
                    "image": image.convert("RGB").resize(
                        (384, 384), Image.Resampling.BILINEAR
                    ),
                }
            )
    for img in [x[1] for x in hist_rgbs] + init_rgbs + ego_rgbs:
        image_pool.append({"type": "image", "image": img})

    if image_goal:
        with Image.open(os.path.join(video_root, nav_goal)) as image:
            image_pool.append(
                {
                    "type": "image",
                    "image": image.convert("RGB").resize(
                        (384, 384), Image.Resampling.BILINEAR
                    ),
                }
            )

    image_pool = deque(image_pool)
    messages = []
    for turn in source:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    content.append(image_pool.popleft())
                else:
                    content.append({"type": "text", "text": seg.strip()})
            messages.append({"role": role, "content": content})

        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    return messages


def replace_image_placeholders_batch(
    input_ids,
    high_res_len,
    anchor_token_id=151652,
    placeholder_token_id=151655,
    target_num=9,
    pad_token_id=151643,
):

    import torch

    batch_size = input_ids.shape[0]
    device = input_ids.device
    processed_seqs = []

    for b_idx in range(batch_size):
        single_input_ids = input_ids[b_idx]

        anchor_positions = (single_input_ids == anchor_token_id).nonzero(as_tuple=True)[
            0
        ]
        if high_res_len:
            anchor_positions = anchor_positions[:-high_res_len]

        new_seq = single_input_ids.clone()

        for anchor_idx in reversed(anchor_positions):
            anchor_idx = anchor_idx.item()

            placeholder_start = anchor_idx + 1
            placeholder_end = placeholder_start
            while (
                placeholder_end < new_seq.shape[0]
                and new_seq[placeholder_end] == placeholder_token_id
            ):
                placeholder_end += 1

            prefix = new_seq[: anchor_idx + 1]
            new_placeholders = torch.tensor(
                [placeholder_token_id] * target_num, dtype=torch.long, device=device
            )
            suffix = new_seq[placeholder_end:]
            new_seq = torch.cat([prefix, new_placeholders, suffix], dim=0)

        processed_seqs.append(new_seq)

    new_input_ids = torch.nn.utils.rnn.pad_sequence(
        sequences=processed_seqs, batch_first=True, padding_value=pad_token_id
    )

    return new_input_ids


def query_qwen(
    model,
    processor,
    tour_video,
    hist_rgbs,
    init_rgbs,
    ego_rgbs,
    video_dir,
    video_root,
    nav_goal,
):
    import torch

    messages = format_item(
        tour_video, hist_rgbs, init_rgbs, ego_rgbs, video_dir, video_root, nav_goal
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    high_res_len = len(init_rgbs) + len(ego_rgbs)
    if str(nav_goal).lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        high_res_len += 1
    img_num = inputs["image_grid_thw"].size(0)
    high_res_mask = torch.zeros(img_num, dtype=torch.bool)
    if high_res_len:
        high_res_mask[-high_res_len:] = True
    inputs["high_res_mask"] = high_res_mask
    inputs["input_ids"] = replace_image_placeholders_batch(
        inputs["input_ids"], high_res_len
    )

    merge_size = model.config.vision_config.spatial_merge_size
    my_grid_thw = inputs["image_grid_thw"].clone()
    my_grid_thw[~high_res_mask, 1:] = 3 * merge_size
    inputs["my_grid_thw"] = my_grid_thw

    inputs["attention_mask"] = inputs["input_ids"].ne(processor.tokenizer.pad_token_id)

    with torch.no_grad():
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        logging.info(output_text)

    try:
        answer_list = output_text[0].split()
        _, frame_idx, _, view_direct, _, u, v, _, stop = answer_list

        frame_idx = int(frame_idx.strip())
        if not 0 <= frame_idx < len(tour_video):
            raise ValueError(f"Invalid tour-video frame index: {frame_idx}")
        direct_to_idx = {"front": 0, "left": 1, "back": 2, "right": 3}
        view_direct = view_direct.strip().lower()
        view_idx = direct_to_idx[view_direct]

        u, v = int(u.strip()), int(v.strip())
        stop = stop.strip().lower()
        if stop not in ["true", "false"]:
            raise ValueError(f"Invalid stop flag: {stop}")
        if u < 0 or u > 384 or v < 0 or v > 384:
            stop = "true"
        return {
            "pred_frame_idx": frame_idx,
            "view_direct": view_direct.strip().lower(),
            "view_idx": view_idx,
            "pixel": (u, v),
            "stop": stop,
        }
    except (ValueError, KeyError, IndexError) as error:
        logging.warning("Invalid model response: %s", error)
        return None
