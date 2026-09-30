from __future__ import annotations

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from main import _anatomy_name, _anatomy_short, _component_fields
from core.games import FruitGames
from core.gestures import Gesture, GestureFrame
from core.model import ComponentMetadata, SceneModel
from core.robotics import RobotController
from core.ui import HUDData, LumenUI


def frame(gesture: Gesture, pointer=None, distance=None, pinch=False) -> GestureFrame:
    return GestureFrame(gesture, pointer, pointer, distance, pinch, (), 0.0, 1.0)


class PlatformTests(unittest.TestCase):
    def test_callout_copy_wraps_to_rendered_pixel_width(self) -> None:
        max_width = 190
        lines = LumenUI._wrap_text_pixels(
            "Executes instructions and calculations without crossing the callout edge",
            max_width,
            0.39,
            max_lines=3,
        )
        self.assertGreater(len(lines), 1)
        for line in lines:
            rendered_width = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                1,
            )[0][0]
            self.assertLessEqual(rendered_width, max_width)

    def test_motherboard_parts_have_structured_educational_fields(self) -> None:
        model = SceneModel.load(*config.MODEL_CATALOG["computer"])
        verified = list(model.components.values())
        self.assertIsNone(model.warning)
        self.assertEqual(len(verified), 58)
        self.assertEqual(len(model.metadata["naming_references"]), 5)
        audit = model.metadata["reference_audit"]
        self.assertEqual(len(audit["reference_4"]), 18)
        self.assertGreaterEqual(len(audit["reference_5"]), 50)
        expected_specific_parts = {
            "eatx12v", "eatxpwr", "am4_socket",
            "dimm_a1", "dimm_a2", "dimm_b1", "dimm_b2",
            "cpu_opt", "cpu_fan", "aio_pump", "w_pump_plus",
            "cha_fan1", "cha_fan2", "cha_fan3", "t_sensor",
            "rgb_header1", "rgb_header2", "cpu_led", "dram_led", "vga_led", "boot_device_led",
            "sata6g_12", "sata6g_34", "sata6g_56", "sata6g_78",
            "pciex16_1", "pciex16_2", "pciex16_3", "pciex1_1", "pciex1_2", "pciex1_3",
            "intel_i211at", "asm1142", "super_io", "supremefx_s1220a", "ics", "tpu",
            "rear_usb12", "rear_hdmi_dp", "rear_usb3_1_4", "rear_usb3_56", "rear_lan_usb31", "rear_audio",
        }
        self.assertTrue(expected_specific_parts.issubset(model.components))
        for component in verified:
            self.assertTrue(component.name.strip(), component.id)
            self.assertTrue(component.short.strip(), component.id)
            self.assertLessEqual(len(component.short), 110, component.id)
            self.assertGreaterEqual(len(component.fields), 2, component.id)
            self.assertIsNotNone(model.target_center(component.id), component.id)
        names = {component.name for component in verified}
        self.assertNotIn("Northbridge", names)
        self.assertNotIn("Southbridge chipset", names)
        self.assertIn("CMOS/BIOS IC / ROM Chip: exact authored mesh cannot be identified reliably", audit["unresolved_in_glb"])

    def test_official_motherboard_hotspots_disambiguate_combined_glb_meshes(self) -> None:
        model = SceneModel.load(*config.MODEL_CATALOG["computer"])
        hotspot_ids = {component_id for component_id, *_bounds in model._hotspots}
        self.assertEqual(len(model._hotspots), 47)
        self.assertEqual(len(hotspot_ids), 46)
        for component_id in hotspot_ids:
            with self.subTest(component_id=component_id):
                center = model.target_center(component_id)
                self.assertIsNotNone(center)
                hit = model.raycast((center[0], center[1], 2.0), (0.0, 0.0, -1.0))
                self.assertIsNotNone(hit)
                self.assertEqual(hit.component_id, component_id)

    def test_stl_uses_the_same_scene_loader(self) -> None:
        model = SceneModel.load(ROOT / "tests" / "fixtures" / "triangle.stl")
        self.assertTrue(model.loaded, model.error)
        self.assertEqual(len(model.primitives), 1)
        self.assertEqual(len(model.primitives[0].triangle_indices) // 3, 1)

    def test_every_catalog_model_loads_and_keeps_primitives(self) -> None:
        expected = {
            "computer": (87, 246757),
            "robotics": (64, 72957),
            "anatomy": (281, 5711437),
            "interactive": (27, 24856),
        }
        for category, (primitive_count, triangle_count) in expected.items():
            with self.subTest(category=category):
                path, metadata = config.MODEL_CATALOG[category]
                model = SceneModel.load(path, metadata)
                self.assertTrue(model.loaded, model.error)
                self.assertEqual(len(model.primitives), primitive_count)
                self.assertEqual(sum(len(item.triangle_indices) // 3 for item in model.primitives), triangle_count)

    def test_original_materials_and_native_lines_are_preserved(self) -> None:
        computer = SceneModel.load(*config.MODEL_CATALOG["computer"])
        fruits = SceneModel.load(*config.MODEL_CATALOG["interactive"])
        anatomy = SceneModel.load(*config.MODEL_CATALOG["anatomy"])
        self.assertGreaterEqual(len({item.material_name for item in computer.primitives}), 20)
        self.assertTrue(any(item.material.texture is not None for item in computer.primitives))
        self.assertEqual(len({item.material_name for item in fruits.primitives}), 14)
        self.assertEqual(sum(len(item.line_indices) // 2 for item in anatomy.primitives), 601)
        self.assertEqual(
            {item.material_name for item in anatomy.primitives if item.primitive_mode == "LINES"},
            {"Muscles.001", "Ligament.002", "Cartilage", "Cartilage.001", "Cartilage.002", "Suture"},
        )
        self.assertEqual(len(anatomy.components), 276)
        muscle = next(
            item
            for item in anatomy.primitives
            if item.material_name == "Muscles.001" and (item.semantic_id or "").startswith("unknown:")
        )
        self.assertEqual(anatomy.component(muscle.semantic_id).name, "MUSCLE")
        self.assertEqual(anatomy.component(muscle.semantic_id).fields["MATERIAL"], "Muscles.001")

    def test_anatomy_card_hides_debug_and_incomplete_metadata(self) -> None:
        complete = ComponentMetadata(
            "pectoralis_major",
            "PECTORALIS MAJOR",
            "Moves the upper arm toward and across the body.",
            "Moves the upper arm toward and across the body.",
            fields={
                "SYSTEM": "Muscular system",
                "LOCATION": "Chest",
                "FUNCTION": "Moves the upper arm toward and across the body.",
                "NODE": "Object_123",
                "MESH": "Mesh_4",
                "MATERIAL": "Muscles.001",
                "RELATION": "Repeats the function",
            },
        )
        no_location = ComponentMetadata(
            "deltoid",
            "DELTOID",
            "Raises and rotates the arm at the shoulder.",
            "Raises and rotates the arm at the shoulder.",
            fields={"SOURCE": "Reference 1", "MODEL PART": "Shoulder region"},
        )
        incomplete = ComponentMetadata(
            "unknown:Object_123",
            "MUSCLE / UNMAPPED STRUCTURE",
            "Requires manual mapping",
            "UNKNOWN",
            fields={
                "NODE": "Object_123",
                "MESH": "Mesh_4",
                "MATERIAL": "Muscles.001",
                "LOCATION": "Requires manual mapping",
            },
        )

        self.assertEqual(
            _component_fields(complete, None, complete.id, "anatomy"),
            (
                ("SYSTEM", "Muscular system"),
                ("LOCATION", "Chest"),
                ("FUNCTION", "Moves the upper arm toward and across the body."),
            ),
        )
        self.assertEqual(
            _component_fields(no_location, None, no_location.id, "anatomy"),
            (
                ("SYSTEM", "Muscular system"),
                ("FUNCTION", "Raises and rotates the arm at the shoulder."),
            ),
        )
        self.assertEqual(_anatomy_name(incomplete), "MUSCLE")
        self.assertEqual(_anatomy_short(incomplete), "")
        incomplete_fields = _component_fields(incomplete, None, incomplete.id, "anatomy")
        self.assertEqual(incomplete_fields, (("SYSTEM", "Muscular system"),))

        anatomy = SceneModel.load(*config.MODEL_CATALOG["anatomy"])
        tendon_primitive = next(item for item in anatomy.primitives if item.material_name == "Tendon.001")
        tendon = anatomy.component(tendon_primitive.semantic_id)
        self.assertEqual(_anatomy_name(tendon), "TENDON")
        self.assertEqual(
            _component_fields(tendon, None, tendon.id, "anatomy"),
            (
                ("SYSTEM", "Musculoskeletal system"),
                ("FUNCTION", "Connects muscle to bone and transfers muscular force."),
            ),
        )

    def test_anatomy_cards_fit_and_wrap_for_required_scenarios(self) -> None:
        ui = LumenUI(False)
        scenarios = (
            HUDData(
                category="anatomy",
                selected_name="PECTORALIS MAJOR",
                selected_fields=(("SYSTEM", "Muscular system"), ("LOCATION", "Chest"), ("FUNCTION", "Moves the upper arm toward and across the body.")),
            ),
            HUDData(
                category="anatomy",
                selected_name="DELTOID",
                selected_fields=(("SYSTEM", "Muscular system"), ("FUNCTION", "Raises and rotates the arm at the shoulder.")),
            ),
            HUDData(
                category="anatomy",
                selected_name="TENDON",
                selected_fields=(("SYSTEM", "Musculoskeletal system"), ("FUNCTION", "Connects muscle to bone and transfers muscular force.")),
            ),
            HUDData(
                category="anatomy",
                selected_name="STERNOCLEIDOMASTOID",
                selected_fields=(("SYSTEM", "Muscular system"), ("LOCATION", "Neck"), ("FUNCTION", "Rotates the head and flexes the neck.")),
            ),
            HUDData(
                category="anatomy",
                selected_name="MUSCLE",
                selected_fields=(("SYSTEM", "Muscular system"), ("LOCATION", "Requires manual mapping"), ("NODE", "Object_123"), ("MESH", "Mesh_4"), ("MATERIAL", "Muscles.001")),
            ),
        )
        forbidden = ("node", "mesh", "material", "object_", "unmapped", "manual mapping", "unknown")
        for data in scenarios:
            with self.subTest(name=data.selected_name):
                layout = ui._selected_card_layout(960, 720, data)
                self.assertGreaterEqual(layout.box[0], 0)
                self.assertGreaterEqual(layout.box[1], 0)
                self.assertLessEqual(layout.box[2], 960)
                self.assertLessEqual(layout.box[3], 720 - 58)
                visible = " ".join(
                    (*layout.title_lines, *layout.detail_lines, *(heading for heading, _ in layout.field_lines), *(line for _, values in layout.field_lines for line in values))
                ).casefold()
                self.assertFalse(any(word in visible for word in forbidden), visible)
                for line in (*layout.title_lines, *layout.detail_lines, *(line for _, values in layout.field_lines for line in values)):
                    rendered_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.57 if line in layout.title_lines else 0.40, 1)[0][0]
                    self.assertLessEqual(rendered_width, layout.box[2] - layout.box[0] - 28)

                y = layout.box[1] + 27 + len(layout.title_lines) * 22
                if layout.detail_lines:
                    y += 2 + len(layout.detail_lines) * 19
                for _heading, values in layout.field_lines:
                    y += 8 + 19 + len(values) * 19 + 4
                self.assertLessEqual(y + 10, layout.box[3])

    def test_all_five_anatomy_references_are_audited(self) -> None:
        model = SceneModel.load(*config.MODEL_CATALOG["anatomy"])
        audit = model.metadata["reference_audit"]
        mapping_audit = model.metadata["mapping_audit"]
        self.assertEqual(len(model.metadata["naming_references"]), 5)
        self.assertEqual(
            [len(audit[f"reference_{index}"]) for index in range(1, 6)],
            [38, 30, 28, 53, 14],
        )
        mapped = [
            component
            for component in model.components.values()
            if not component.id.startswith("unknown:")
        ]
        self.assertEqual(len(mapped), 9)
        self.assertEqual(len(mapping_audit["validated"]), 9)
        self.assertTrue(mapping_audit["retired"])
        self.assertEqual(
            {entry["status"] for entry in mapping_audit["validated"]},
            {"CORRECT"},
        )
        self.assertLessEqual(
            {entry["status"] for entry in mapping_audit["retired"]},
            {"PARTIALLY CORRECT", "INCORRECT", "AMBIGUOUS", "UNMAPPED"},
        )
        for component in mapped:
            with self.subTest(component_id=component.id):
                self.assertTrue(component.name.strip())
                self.assertTrue(component.short.strip())
                self.assertLessEqual(len(component.short), 110)
                self.assertFalse(model.component_regions(component.id))
                self.assertIsNotNone(model.target_center(component.id))
                self.assertEqual(set(component.fields), {"SYSTEM", "LOCATION", "FUNCTION"})

    def test_anatomy_uses_only_audited_exact_nodes_for_pointing(self) -> None:
        model = SceneModel.load(*config.MODEL_CATALOG["anatomy"])
        validated = model.metadata["mapping_audit"]["validated"]
        expected = {entry["id"]: set(entry["nodes"]) for entry in validated}
        actual: dict[str, set[str]] = {component_id: set() for component_id in expected}
        owner_by_node: dict[str, str] = {}
        for primitive in model.primitives:
            if primitive.semantic_id not in actual:
                continue
            actual[primitive.semantic_id].add(primitive.node_name)
            previous = owner_by_node.setdefault(primitive.node_name, primitive.semantic_id)
            self.assertEqual(previous, primitive.semantic_id)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["latissimus_dorsi_right"], {"Object_152"})
        self.assertNotIn("Object_115", owner_by_node)  # mixed gluteal/head geometry
        self.assertNotIn("Object_161", owner_by_node)  # mixed calf/non-calf geometry
        self.assertTrue(all(not model.component_regions(component_id) for component_id in expected))

        # Probe authored triangle surfaces directly.  This confirms that every
        # audited node can actually win normal pointing/raycast selection,
        # including deep structures that are occluded at their bounding-box centre.
        selectable_nodes: set[str] = set()
        for primitive in model.primitives:
            if primitive.semantic_id not in expected:
                continue
            triangles = np.stack(
                (
                    primitive.triangle_v0,
                    primitive.triangle_v0 + primitive.triangle_edge1,
                    primitive.triangle_v0 + primitive.triangle_edge2,
                ),
                axis=1,
            )
            indices = np.linspace(0, len(triangles) - 1, min(80, len(triangles)), dtype=np.int64)
            local = triangles[indices].reshape(-1, 3)
            homogeneous = np.column_stack((local, np.ones(len(local))))
            world = (homogeneous @ model.node_matrix(primitive).T)[:, :3].reshape(-1, 3, 3)
            for triangle in world:
                normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                centre = triangle.mean(axis=0)
                for sign in (-1.0, 1.0):
                    hit = model.raycast(centre + normal * sign * 0.01, -normal * sign)
                    if (
                        hit is not None
                        and hit.component_id == primitive.semantic_id
                        and hit.node_name == primitive.node_name
                    ):
                        selectable_nodes.add(primitive.node_name)
                        break
                if primitive.node_name in selectable_nodes:
                    break
        self.assertEqual(selectable_nodes, set(owner_by_node))

    def test_fruits_are_seven_independent_logical_objects(self) -> None:
        model = SceneModel.load(*config.MODEL_CATALOG["interactive"])
        objects = {item.object_id for item in model.primitives}
        self.assertEqual(objects, {"red_apple", "green_apple", "banana_1", "banana_2", "banana_3", "orange", "pear"})
        orange_nodes = {item.node_name for item in model.primitives if item.object_id == "orange"}
        pear_nodes = {item.node_name for item in model.primitives if item.object_id == "pear"}
        self.assertIn("Leaf.003_Material_0", orange_nodes)
        self.assertNotIn("Leaf.003_Material_0", pear_nodes)

    def test_robot_uses_legacy_joint_groups_limits_and_smoothing(self) -> None:
        model = SceneModel.load(*config.MODEL_CATALOG["robotics"])
        robot = RobotController(model)
        self.assertEqual([len(robot.joints[name].nodes) for name in robot.order], [4, 12, 8, 7, 30])
        self.assertEqual(robot.limits("base"), (-170.0, 170.0))
        self.assertEqual(robot.limits("elbow"), (-135.0, 20.0))
        self.assertLess(max(np.max(np.abs(value - np.eye(4))) for value in model._deformations.values()), 1e-8)
        before = np.asarray(model.target_center("elbow"))
        robot.add_drag("elbow", 0.0, 1.0)
        robot.update(1.0 / 60.0)
        self.assertGreater(robot.angle("elbow"), -2.0)
        self.assertLess(robot.angle("elbow"), 0.0)
        for _ in range(180):
            robot.update(1.0 / 60.0)
        self.assertAlmostEqual(robot.angle("elbow"), -132.0, places=3)
        self.assertEqual(len(model._deformations), 61)
        gripper_node = next(
            item.node_name for item in model.primitives
            if item.display_name == "roboarm.002_low_1"
        )
        self.assertIn(gripper_node, model._deformations)
        self.assertEqual(robot.node_to_joint[gripper_node], "wrist_roll")
        self.assertGreater(np.linalg.norm(np.asarray(model.target_center("elbow")) - before), 0.01)

    def test_fruit_match_and_size_use_object_transforms(self) -> None:
        model = SceneModel.load(*config.MODEL_CATALOG["interactive"])
        games = FruitGames(model, FruitGames.MATCH)
        self.assertTrue(all(box.rect[1] >= 0.7 for box in games.boxes))
        self.assertEqual(len({box.color_rgb for box in games.boxes}), 4)
        target = next(box for box in games.boxes if box.kind == "pear")
        center = ((target.rect[0] + target.rect[2]) * 0.5, (target.rect[1] + target.rect[3]) * 0.5)
        games.update(frame(Gesture.PINCH, (0.5, 0.5), pinch=True), "pear", 1.0)
        games.update(frame(Gesture.PINCH, center, pinch=True), "pear", 1.0)
        games.update(frame(Gesture.POINT, center, pinch=False), "pear", 1.0)
        self.assertIn("pear", games.matched)
        self.assertIn("pear", model._object_deformations)

        games.set_mode(FruitGames.SIZE)
        games.update(frame(Gesture.TWO_HANDS, distance=0.2), "pear", 1.0)
        games.update(frame(Gesture.TWO_HANDS, distance=0.33), "pear", 1.0)
        self.assertTrue(games.dimension_match)
        self.assertTrue(np.isfinite(model._object_deformations["pear"]).all())

    def test_both_fruit_games_allow_individual_move_rotate_and_scale(self) -> None:
        for mode in (FruitGames.MATCH, FruitGames.SIZE):
            model = SceneModel.load(*config.MODEL_CATALOG["interactive"])
            games = FruitGames(model, mode)
            object_id = "pear"

            games.update(frame(Gesture.PINCH, (0.40, 0.40), pinch=True), object_id, 1.0)
            games.update(frame(Gesture.PINCH, (0.48, 0.46), pinch=True), object_id, 1.0)
            games.update(frame(Gesture.POINT, (0.48, 0.46)), object_id, 1.0)

            games.update(frame(Gesture.V_SIGN, (0.42, 0.42)), object_id, 1.0)
            games.update(frame(Gesture.V_SIGN, (0.55, 0.50)), object_id, 1.0)

            games.update(frame(Gesture.TWO_HANDS, distance=0.20), object_id, 1.0)
            games.update(frame(Gesture.TWO_HANDS, distance=0.30), object_id, 1.0)

            self.assertGreater(np.linalg.norm(games.object_offsets[object_id]), 0.01, mode)
            self.assertGreater(np.linalg.norm(games.object_rotations[object_id]), 1.0, mode)
            self.assertGreater(games.object_scales[object_id], 1.0, mode)
            self.assertTrue(np.isfinite(model._object_deformations[object_id]).all(), mode)

    def test_fruit_target_boxes_render_with_their_own_colors(self) -> None:
        model = SceneModel.load(*config.MODEL_CATALOG["interactive"])
        games = FruitGames(model, FruitGames.MATCH)
        boxes = tuple((box.label, *box.rect, False, box.color_rgb) for box in games.boxes)
        overlay = LumenUI(False).draw(960, 720, HUDData(game_mode="match", game_boxes=boxes))
        for box in games.boxes:
            x = int((box.rect[0] * 960) + 10)
            y = int((box.rect[1] * 720) + 10)
            expected_bgr = np.asarray(box.color_rgb[::-1], dtype=np.float64)
            actual_bgr = overlay[y, x, :3].astype(np.float64)
            self.assertGreater(float(np.dot(actual_bgr, expected_bgr)), 0.0, box.label)
            self.assertGreater(int(overlay[y, x, 3]), 0, box.label)


if __name__ == "__main__":
    unittest.main()
