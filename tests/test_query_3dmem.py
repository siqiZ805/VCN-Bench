"""Object-selection contracts; optional real processor check uses a local checkpoint."""

import json
import os
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from model import query_3dmem as adapter


@dataclass
class Snapshot:
    image: str
    obs_point: object
    full_obj_list: dict = field(default_factory=dict)
    cluster: list = field(default_factory=list)


class ObjectSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        Image.new("RGB", (80, 40), "red").save(self.root / "frame.png")
        Image.new("RGB", (80, 40), "blue").save(self.root / "goal.png")
        self.cfg = SimpleNamespace(
            prompt_w=64, prompt_h=64, use_full_obj_list=False,
            egocentric_views=True, prefiltering=False, top_k_categories=10,
            video_root=str(self.root),
        )
        objects = {11: {"class_name": "table", "conf": 0.8},
                   42: {"class_name": "chair", "conf": 0.9},
                   73: {"class_name": "chair", "conf": 0.7}}
        old = Snapshot("1-view_0.png", np.array([1, 2, 3]), {11: 0.8, 42: 0.9, 73: 0.7}, [11, 42, 73])
        old.visual_prompt = SimpleNamespace(
            xyxy=np.array([[0, 0, 20, 20], [-2, 2, 40, 22], [40, 2, 80, 22]]),
            data={"obj_id": np.array([11, 42, 73])},
        )
        current = Snapshot("5-view_0.png", np.array([4, 5, 6]), {11: 0.8}, [11])
        current.visual_prompt = SimpleNamespace(xyxy=np.array([[0, 0, 20, 20]]), data={"obj_id": [11]})
        self.scene = SimpleNamespace(
            objects=objects, snapshots={old.image: old, current.image: current},
            all_observations={old.image: np.full((40, 80, 4), 255, dtype=np.uint8),
                              current.image: np.zeros((40, 80, 3), dtype=np.uint8)},
        )
        frontier = SimpleNamespace(image="frontier.png", feature=np.zeros((32, 32, 3), dtype=np.uint8))
        self.planner = SimpleNamespace(frontiers=[frontier])

    def prepare(self, goal="Find the chair in the second room"):
        return adapter.prepare_vlm_input_dict(
            {"goal": goal, "goal_obj_ids": [999], "gt_bbox": [0] * 6},
            self.scene, self.planner, [np.zeros((40, 80, 3), dtype=np.uint8)],
            ["frame.png"], str(self.root), self.cfg, global_step=5,
        )

    def test_history_and_current_snapshots_are_both_selectable(self):
        inputs = self.prepare()
        self.assertEqual([s["image"] for s in inputs["snapshots"]], ["1-view_0.png", "5-view_0.png"])
        selection = adapter.parse_selection("Frame: 0. Answer: Snapshot 0, Object 2", inputs)
        self.assertEqual(selection["pred_obj_id"], 73)
        self.assertEqual(selection["pred_snapshot_image"], "1-view_0.png")
        self.assertNotIn(999, inputs["obj_map"])

    def test_crop_clipping_aspect_ratio_invalid_and_deleted_ids(self):
        old = self.scene.snapshots["1-view_0.png"]
        old.visual_prompt.xyxy = np.array([
            [-2, 2, 40, 22], [80, 0, 85, 10], [0, 0, 20, 20],
            [np.nan, 0, 20, 20], [0, 0, 10, 10],
        ])
        old.visual_prompt.data["obj_id"] = [42, 11, 999, 73, 42]
        old.cluster.append(999)
        objs = self.prepare()["snapshots"][0]["objects"]
        self.assertEqual([obj["obj_id"] for obj in objs], [42])
        self.assertEqual(objs[0]["bbox"], [0, 2, 40, 22])
        self.assertEqual(objs[0]["crop"].size, (40, 20))
        self.assertEqual(objs[0]["crop"].mode, "RGB")

    def test_full_obj_list_controls_candidates(self):
        self.scene.snapshots["1-view_0.png"].cluster = [11]
        self.assertEqual(len(self.prepare()["snapshots"][0]["objects"]), 1)
        self.cfg.use_full_obj_list = True
        self.assertEqual(len(self.prepare()["snapshots"][0]["objects"]), 3)

    def test_missing_detections_and_empty_snapshots_are_not_candidates(self):
        del self.scene.snapshots["1-view_0.png"].visual_prompt
        self.scene.snapshots["5-view_0.png"].cluster = []
        self.assertEqual(self.prepare()["snapshots"], [])

    def test_detection_mapping_mismatch_is_explicit(self):
        self.scene.snapshots["1-view_0.png"].visual_prompt.data["obj_id"] = []
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            self.prepare()

    def test_message_labels_are_adjacent_to_the_correct_images(self):
        inputs = self.prepare("goal.png")
        messages = adapter.format_item(inputs)
        self.assertEqual([msg["role"] for msg in messages], ["system", "user"])
        content = messages[1]["content"]
        pairs = [(content[i - 1]["text"], item["image"])
                 for i, item in enumerate(content) if item["type"] == "image"]
        self.assertEqual(pairs[0][0], "Navigation Goal (target image):")
        self.assertEqual(pairs[0][1].getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(pairs[1][0], "Frame 0:")
        self.assertEqual(pairs[1][1].getpixel((0, 0)), (255, 0, 0))
        self.assertIs(next(image for label, image in pairs if label == "Snapshot 0 (full image):"),
                      inputs["snapshots"][0]["full_img"])
        self.assertEqual([label for label, _ in pairs if label.startswith("Object")],
                         ["Object 0: table", "Object 1: chair", "Object 2: chair", "Object 0: table"])

    def test_initial_directions_preserve_the_episode_start_reference(self):
        for index in range(4):
            self.scene.all_observations[f"0-view_{index}.png"] = np.full(
                (40, 80, 3), index * 40, dtype=np.uint8,
            )
        messages = adapter.format_item(self.prepare())
        content = messages[1]["content"]
        pairs = [(content[index - 1]["text"], item["image"])
                 for index, item in enumerate(content) if item["type"] == "image"
                 and content[index - 1]["text"].startswith("Initial View")]
        self.assertEqual([label for label, _ in pairs], [
            "Initial View 0 (front at episode start):", "Initial View 1 (left at episode start):",
            "Initial View 2 (back at episode start):", "Initial View 3 (right at episode start):",
        ])
        self.assertEqual([image.getpixel((0, 0))[0] for _, image in pairs], [120, 80, 0, 40])

    def test_invalid_formats_and_indices_are_rejected(self):
        inputs = self.prepare()
        cases = ["Frame: 0. Answer: Snapshot 0", "Frame: 0. Answer: Snapshot 0, Object -1",
                 "Frame: 1. Answer: Snapshot 0, Object 1", "Frame: 0. Answer: Snapshot 2, Object 0",
                 "Frame: 0. Answer: Snapshot 0, Object 3", "Frame: 0. Answer: Frontier 1",
                 "Frame: 0. Answer: Frontier 0, Object 1", "Frame: 0. Answer: Frontier 0\nexplanation"]
        for response in cases:
            with self.subTest(response=response), self.assertRaises(ValueError):
                adapter.parse_selection(response, inputs)

    def test_same_class_instances_remain_distinct_singleton_targets(self):
        original = self.scene.snapshots["1-view_0.png"]
        for index, object_id in [(1, 42), (2, 73)]:
            selection = adapter.parse_selection(f"Frame: 0. Answer: Snapshot 0, Object {index}", self.prepare())
            choice = adapter.navigation_choice(selection, self.scene, self.planner)
            self.assertIsNot(choice, original)
            self.assertEqual(choice.cluster, [object_id])
            self.assertEqual(choice.full_obj_list, {object_id: self.scene.objects[object_id]["conf"]})
            choice.cluster.append(-1)
            choice.obs_point[0] = -1
        self.assertEqual(original.cluster, [11, 42, 73])
        np.testing.assert_array_equal(original.obs_point, [1, 2, 3])

    def test_deleted_object_is_not_used_for_planning(self):
        selection = adapter.parse_selection("Frame: 0. Answer: Snapshot 0, Object 1", self.prepare())
        del self.scene.objects[42]
        with self.assertRaisesRegex(ValueError, "no longer exists"):
            adapter.navigation_choice(selection, self.scene, self.planner)

    def test_frontier_has_no_object_and_preserves_frontier_identity(self):
        selection = adapter.parse_selection("Frame: 0. Answer: Frontier 0", self.prepare())
        self.assertIsNone(selection["pred_obj_id"])
        self.assertIsNone(selection["pred_obj_idx"])
        self.assertIsNone(selection["pred_snapshot_image"])
        self.assertIs(adapter.navigation_choice(selection, self.scene, self.planner), self.planner.frontiers[0])

    def test_prefilter_renumbers_snapshot_and_object_indices(self):
        self.cfg.prefiltering = True
        inputs = self.prepare()
        # Move the chair snapshot behind the table snapshot to exercise both mappings.
        inputs["snapshots"].reverse()
        with patch.object(adapter, "_generate", side_effect=["chair", "Frame: 0. Answer: Snapshot 0, Object 1"]):
            record, choice = adapter.query_qwen(inputs, self.cfg, self.scene, self.planner, None, None)
        self.assertEqual(choice.cluster, [73])
        self.assertEqual(record["pred_snap_idx"], 0)
        self.assertEqual(record["pred_obj_idx"], 1)
        self.assertEqual(record["selected_categories"], ["chair"])
        snapshots = record["candidate_mapping"]["snapshots"]
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["image"], "1-view_0.png")
        self.assertEqual([obj["obj_id"] for obj in snapshots[0]["objects"]], [42, 73])
        json.dumps(record)
        self.assertEqual(len(inputs["snapshots"]), 2)

    def test_retries_include_validation_error_and_audit_responses(self):
        with patch.object(adapter, "_generate", side_effect=[
            "Frame: 0. Answer: Snapshot 0", "Frame: 0. Answer: Snapshot 0, Object 1",
        ]) as generate:
            record, choice = adapter.query_qwen(self.prepare(), self.cfg, self.scene, self.planner, None, None)
        self.assertEqual(choice.cluster, [42])
        self.assertEqual(len(record["responses"]), 2)
        self.assertIn("Invalid response", generate.call_args_list[1].args[0][-1]["content"][0]["text"])
        self.assertIsNone(record["error"])

    def test_format_failure_never_falls_back_to_a_cluster_or_random_object(self):
        with patch.object(adapter, "_generate", return_value="Frame: 0. Answer: Snapshot 0") as generate:
            with self.assertLogs(level="ERROR"):
                record, choice = adapter.query_qwen(self.prepare(), self.cfg, self.scene, self.planner, None, None)
        self.assertIsNone(choice)
        self.assertEqual(generate.call_count, 3)
        self.assertIn("three attempts", record["error"])
        self.assertEqual(len(record["responses"]), 3)

    def test_prefilter_none_leaves_only_frontiers(self):
        self.cfg.prefiltering = True
        with patch.object(adapter, "_generate", side_effect=["NONE", "Frame: 0. Answer: Frontier 0"]):
            record, choice = adapter.query_qwen(self.prepare(), self.cfg, self.scene, self.planner, None, None)
        self.assertIs(choice, self.planner.frontiers[0])
        self.assertEqual(record["candidate_mapping"]["snapshots"], [])

    def test_invalid_prefilter_output_is_retried(self):
        self.cfg.prefiltering = True
        self.cfg.top_k_categories = 1
        with patch.object(adapter, "_generate", side_effect=[
            "chair\ntable", "invented_category", "chair", "Frame: 0. Answer: Snapshot 0, Object 0",
        ]):
            record, choice = adapter.query_qwen(self.prepare(), self.cfg, self.scene, self.planner, None, None)
        self.assertEqual(choice.cluster, [42])
        self.assertEqual(len(record["prefilter_responses"]), 3)

    def test_runtime_failure_is_recorded_without_retry_or_truncation(self):
        with patch.object(adapter, "_generate", side_effect=RuntimeError("CUDA out of memory")) as generate:
            with self.assertLogs(level="ERROR"):
                record, choice = adapter.query_qwen(self.prepare(), self.cfg, self.scene, self.planner, None, None)
        self.assertIsNone(choice)
        self.assertEqual(generate.call_count, 1)
        self.assertIn("CUDA out of memory", record["error"])
        self.assertEqual(len(record["candidate_mapping"]["snapshots"]), 2)

    def test_no_candidates_or_no_video_is_explicit(self):
        inputs = self.prepare()
        for key in ["candidates", "video"]:
            empty = {**inputs, "snapshots": [], "frontier_imgs": [], "frontier_names": []} if key == "candidates" else {**inputs, "tour_video": []}
            with self.subTest(key=key), patch.object(adapter, "_generate") as generate, self.assertLogs(level="ERROR"):
                record, choice = adapter.query_qwen(empty, self.cfg, self.scene, self.planner, None, None)
                self.assertIsNone(choice)
                self.assertIsNotNone(record["error"])
                generate.assert_not_called()

    def test_checkpoint_config_and_protocol_control_result_reuse(self):
        config = {"qwen_path": "Qwen/Qwen3-VL-4B-Instruct", "prefiltering": True, "exp_name": "one"}
        identity = adapter.evaluation_identity(config)
        result = {**identity, "evaluation_status": "completed"}
        self.assertTrue(adapter.result_is_reusable(result, identity))
        self.assertFalse(adapter.result_is_reusable({**result, "evaluation_status": "error"}, identity))
        with self.assertRaises(ValueError):
            adapter.result_is_reusable({}, identity)
        for change in [{"qwen_path": "Qwen/Qwen3-VL-8B-Instruct"}, {"prefiltering": False}]:
            with self.assertRaises(ValueError):
                adapter.result_is_reusable(result, adapter.evaluation_identity({**config, **change}))
        self.assertEqual(identity, adapter.evaluation_identity({**config, "exp_name": "two"}))

    def test_model_loader_uses_standard_class_and_matching_processor_for_both_sizes(self):
        factory = unittest.mock.Mock()
        processor = unittest.mock.Mock()
        fake_transformers = SimpleNamespace(Qwen3VLForConditionalGeneration=factory, AutoProcessor=processor)
        with patch.dict("sys.modules", {"torch": SimpleNamespace(bfloat16="bfloat16"), "transformers": fake_transformers}):
            for size in ["4B", "8B"]:
                name = f"Qwen/Qwen3-VL-{size}-Instruct"
                adapter.build_model(name)
                self.assertEqual(factory.from_pretrained.call_args.args, (name,))
                processor.from_pretrained.assert_called_with(name)
            adapter.build_model("local-model", "local-processor")
            processor.from_pretrained.assert_called_with("local-processor")

    @unittest.skipUnless(os.environ.get("VCN_QWEN_PROCESSOR"), "Set VCN_QWEN_PROCESSOR to a local original checkpoint")
    def test_real_standard_processor_preserves_visual_tokens_and_context_limit(self):
        import torch
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(os.environ["VCN_QWEN_PROCESSOR"], local_files_only=True)
        inputs = self.prepare()
        inputs["image_size"] = (384, 384)
        messages = adapter.format_item(inputs)
        batch = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", truncation=False,
        )
        grid = batch["image_grid_thw"]
        image_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        expected = (grid.prod(-1) // processor.image_processor.merge_size ** 2).sum().item()
        self.assertEqual((batch["input_ids"] == image_token).sum().item(), expected)
        self.assertGreater((grid[0].prod() // processor.image_processor.merge_size ** 2).item(), 9)
        self.assertNotIn("high_res_mask", batch)
        self.assertNotIn("my_grid_thw", batch)
        self.assertEqual(len(grid), sum(item["type"] == "image" for item in messages[1]["content"]))

        answer = "Frame: 0. Answer: Snapshot 0, Object 1"
        generated_answer = processor.tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt")
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=262144), device="cpu")

        def generate(**kwargs):
            self.assertNotIn("high_res_mask", kwargs)
            self.assertNotIn("my_grid_thw", kwargs)
            self.assertFalse(kwargs["do_sample"])
            return torch.cat([kwargs["input_ids"], generated_answer], dim=-1)

        model.generate = unittest.mock.Mock(side_effect=generate)
        self.assertEqual(adapter._generate(messages, model, processor, 64), answer)
        model.config.max_position_embeddings = 8
        model.generate.reset_mock()
        with self.assertRaisesRegex(ValueError, "No inputs were truncated"):
            adapter._generate(messages, model, processor, 64)
        model.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
