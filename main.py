"""Lumen application entry point: one camera, renderer, gesture system, and UI."""

from __future__ import annotations

from dataclasses import replace
import sys
import time

import glfw

import config
from ai.assistant import JeremyAssistant
from ai.context import LumenContextBuilder
from ai.settings import JeremySettings
from core.camera import CameraCapture
from core.games import FruitGames
from core.gestures import Gesture, GestureFrame, GestureRecognizer
from core.interaction import InputMode, InteractionController, InteractionState
from core.model import ComponentMetadata, RaycastHit, SceneModel
from core.renderer import Renderer
from core.robotics import RobotController
from core.tracking import HandTracker, TrackingSnapshot
from core.ui import HUDData, LumenUI, UICommand


class _RateCounter:
    def __init__(self, interval: float = 0.6) -> None:
        self.value = 0.0
        self._interval = interval
        self._started = time.perf_counter()
        self._last_tick = self._started
        self._count = 0

    def tick(self, timestamp: float) -> None:
        self._last_tick = timestamp
        self._count += 1
        elapsed = timestamp - self._started
        if elapsed >= self._interval:
            self.value = self._count / max(elapsed, 1e-6)
            self._count = 0
            self._started = timestamp

    def displayed(self, timestamp: float) -> float:
        return 0.0 if timestamp - self._last_tick > max(self._interval * 1.5, 0.8) else self.value


class _KeyEdges:
    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer
        self._previous: dict[int, bool] = {}

    def pressed(self, key: int) -> bool:
        down = self._renderer.key_down(key)
        rising = down and not self._previous.get(key, False)
        self._previous[key] = down
        return rising


def _empty_gesture(timestamp: float) -> GestureFrame:
    return GestureFrame(Gesture.NONE, None, None, None, False, (), timestamp, 0.0)


def _fresh_snapshot(snapshot: TrackingSnapshot, timestamp: float) -> TrackingSnapshot:
    if snapshot.hands and snapshot.received_at > 0.0 and timestamp - snapshot.received_at > config.TRACK_LOST_RESET_SECONDS:
        return replace(snapshot, hands=(), status="TRACK LOST")
    return snapshot


def _instruction(
    state: InteractionState,
    mode: InputMode,
    gesture: Gesture,
    hovered: str | None,
    selected: str | None,
    tutorial_visible: bool,
    models_open: bool,
    hand_count: int,
    category: str,
    game_mode: str,
) -> str:
    if tutorial_visible:
        return ""
    if models_open:
        return "CATEGORIES  /  POINT + PINCH TO SELECT"
    if state == InteractionState.STOPPED:
        return "STOPPED  /  OPEN PALM OR ENTER TO RESUME"
    if category == "robotics" and gesture == Gesture.ROCK_SIGN:
        return "HOLD ROCK SIGN  /  TOGGLE ROBOT DEMONSTRATION"
    if category == "robotics" and selected:
        return "PINCH + DRAG  /  MOVE SELECTED JOINT  /  X TO CLEAR"
    if game_mode == FruitGames.SIZE:
        return "PEAR: PINCH MOVE  /  V ROTATE  /  TWO HANDS SCALE"
    if game_mode == FruitGames.MATCH and selected:
        return "PINCH MOVE  /  V ROTATE  /  TWO HANDS SCALE"
    if game_mode == FruitGames.MATCH:
        return "PINCH A FRUIT  /  MOVE IT TO THE MATCHING BOX"
    if mode == InputMode.MOUSE:
        return "MOUSE  /  DRAG TO ROTATE  /  WHEEL TO SCALE"
    if state == InteractionState.MOVE:
        return "OPEN PALM  /  MOVE MODEL"
    if state == InteractionState.ROTATE:
        return "V  /  ROTATE MODEL"
    if state == InteractionState.SCALE:
        return "TWO HANDS  /  SCALE MODEL"
    if state == InteractionState.PINCH_PENDING:
        return "HOLD PINCH  /  CONFIRM"
    if selected:
        return "COMPONENT SELECTED  /  X TO CLEAR"
    if hovered:
        return "PINCH + HOLD  /  TELL ME MORE"
    if gesture == Gesture.NONE and hand_count == 0:
        return "SHOW YOUR HAND  /  POINT AT A PART"
    return "POINT AT A PART"


def _message_parts(message: str | None, fallback: str) -> tuple[str, str]:
    lines = [line.strip() for line in str(message or "").splitlines() if line.strip()]
    return ("", "") if not lines else (lines[0] or fallback, "  /  ".join(lines[1:]))


def _visible_error(model: SceneModel, camera: CameraCapture, tracker: HandTracker) -> tuple[str, str]:
    if model.error:
        return _message_parts(model.error, "MODEL LOAD ERROR")
    if tracker.error:
        return "HAND TRACKING ERROR", tracker.error
    if camera.error:
        return "CAMERA UNAVAILABLE", camera.error
    if model.warning:
        return _message_parts(model.warning, "MODEL WARNING")
    return "", ""


def _debug_lines(
    *,
    render_rate: float,
    tracking_rate: float,
    snapshot: TrackingSnapshot,
    gesture: GestureFrame,
    controller: InteractionController,
    hit: RaycastHit | None,
    raycast_ms: float,
    camera: CameraCapture,
    model: SceneModel,
) -> tuple[str, ...]:
    state = controller.state
    pointer = "-" if gesture.pointer is None else f"{gesture.pointer[0]:.3f}, {gesture.pointer[1]:.3f}"
    return (
        f"RENDER FPS       {render_rate:6.1f}",
        f"TRACKING FPS     {tracking_rate:6.1f}",
        f"TRACK LATENCY    {snapshot.latency_ms:6.1f} ms",
        f"GESTURE          {gesture.gesture.value}",
        f"POINTER          {pointer}",
        f"RAYCAST TARGET   {'-' if hit is None else hit.component_id or '-'}",
        f"NODE             {'-' if hit is None else hit.node_name}",
        f"MESH             {'-' if hit is None else hit.mesh_name}",
        f"MATERIAL         {'-' if hit is None else hit.material_name}",
        f"HOVER            {state.hovered_component or '-'}",
        f"SELECTED         {state.selected_component or '-'}",
        f"RAYCAST TIME     {raycast_ms:6.2f} ms",
        f"PRIMITIVES       {len(model.primitives)}",
        f"STATE            {state.interaction.value}",
        f"CAMERA           {camera.status}",
    )


def _save_tutorial_state() -> None:
    try:
        config.TUTORIAL_STATE_PATH.write_text("completed\n", encoding="utf-8")
    except OSError:
        pass


def _handle_keys(keys: _KeyEdges, renderer: Renderer, ui: LumenUI, controller: InteractionController) -> None:
    if keys.pressed(glfw.KEY_ESCAPE) or keys.pressed(glfw.KEY_Q):
        renderer.request_close()
    if keys.pressed(glfw.KEY_H):
        ui.show_tutorial()
    if keys.pressed(glfw.KEY_D):
        controller.state.debug = not controller.state.debug
    if keys.pressed(glfw.KEY_R):
        controller.reset_view()
    if keys.pressed(glfw.KEY_X):
        controller.clear_selection()
    if keys.pressed(glfw.KEY_SPACE):
        controller.stop("keyboard stop")
    if keys.pressed(glfw.KEY_ENTER) or keys.pressed(glfw.KEY_KP_ENTER):
        controller.resume("keyboard resume")
    if keys.pressed(glfw.KEY_B):
        controller.toggle_input_mode()


def _load_category(category: str) -> SceneModel:
    model_path, metadata_path = config.MODEL_CATALOG[category]
    return SceneModel.load(model_path, metadata_path)


_ANATOMY_BLOCKED_TEXT = (
    "unmapped",
    "unknown",
    "requires manual mapping",
    "object_",
    "mesh_",
    "null",
    "n/a",
)


def _clean_anatomy_text(value: object) -> str:
    text = str(value or "").strip()
    lowered = text.casefold()
    return "" if not text or any(marker in lowered for marker in _ANATOMY_BLOCKED_TEXT) else text


def _anatomy_name(metadata: ComponentMetadata | None) -> str:
    if metadata is None:
        return ""
    name = _clean_anatomy_text(metadata.name)
    if name:
        return name
    material = str(metadata.fields.get("MATERIAL", "")).casefold()
    if "tendon" in material:
        return "TENDON"
    if "muscle" in material:
        return "MUSCLE"
    if "bone" in material:
        return "BONE"
    if "ligament" in material:
        return "LIGAMENT"
    if "cartilage" in material:
        return "CARTILAGE"
    return "ANATOMICAL STRUCTURE"


def _anatomy_short(metadata: ComponentMetadata | None) -> str:
    if metadata is None:
        return ""
    return _clean_anatomy_text(metadata.short) or _clean_anatomy_text(metadata.description)


def _component_fields(
    metadata: ComponentMetadata | None,
    robot: RobotController | None,
    component_id: str | None,
    category: str = "",
) -> tuple[tuple[str, str], ...]:
    if metadata is None:
        return ()
    if category == "anatomy":
        source = metadata.fields
        system = _clean_anatomy_text(source.get("SYSTEM")) or "Muscular system"
        location = _clean_anatomy_text(source.get("LOCATION"))
        function = _clean_anatomy_text(source.get("FUNCTION")) or _anatomy_short(metadata)
        return tuple(
            (heading, value)
            for heading, value in (
                ("SYSTEM", system),
                ("LOCATION", location),
                ("FUNCTION", function),
            )
            if value
        )
    fields = list(metadata.fields.items())
    if robot is not None:
        angle = robot.angle(component_id)
        limits = robot.limits(component_id)
        if angle is not None:
            fields.append(("CURRENT ANGLE", f"{angle:.1f}°"))
        if limits is not None:
            fields.append(("JOINT LIMITS", f"{limits[0]:g}° to {limits[1]:g}°"))
        if component_id == "gripper":
            fields.append(("OPEN", f"{robot.gripper_open * 100:.0f}%"))
    return tuple(fields)


def run() -> int:
    category = "computer"
    game_mode = ""
    model = _load_category(category)
    renderer: Renderer | None = None
    camera: CameraCapture | None = None
    tracker: HandTracker | None = None
    jeremy: JeremyAssistant | None = None

    try:
        renderer = Renderer(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT, model)
        camera = CameraCapture(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS)
        tracker = HandTracker(config.HAND_LANDMARKER_PATH, max_hands=2)
        recognizer = GestureRecognizer()
        controller = InteractionController(model.default_scale, model.default_rotation)
        robot: RobotController | None = None
        fruits: FruitGames | None = None
        ui = LumenUI(tutorial_visible=not config.TUTORIAL_STATE_PATH.is_file())
        ui.set_active_model(category)
        keys = _KeyEdges(renderer)
        context_builder = LumenContextBuilder()
        jeremy = JeremyAssistant(JeremySettings.load(config.BASE_DIR))
        jeremy.start()

        render_rate, tracking_rate = _RateCounter(), _RateCounter()
        previous_time = time.perf_counter()
        previous_tracking_sequence = -1
        processed_tracking_token: tuple[int, str, bool] | None = None
        previous_input_mode = controller.state.input_mode
        submitted_camera_sequence = -1
        raw_hit: RaycastHit | None = None
        gesture_frame = _empty_gesture(previous_time)
        robot_drag_pointer: tuple[float, float] | None = None
        rock_since: float | None = None
        rock_latched = False

        def activate(value: str) -> None:
            nonlocal category, game_mode, model, controller, robot, fruits
            nonlocal previous_input_mode, processed_tracking_token, raw_hit, robot_drag_pointer
            requested_category, _, requested_game = value.partition(":")
            requested_category = requested_category or "computer"
            if requested_category not in config.MODEL_CATALOG:
                return
            if requested_category == category and requested_category == "interactive":
                game_mode = requested_game or game_mode or FruitGames.MATCH
                if fruits is not None:
                    fruits.set_mode(game_mode)
                return
            old_mode = controller.state.input_mode
            old_debug = controller.state.debug
            new_model = _load_category(requested_category)
            renderer.set_model(new_model)
            model = new_model
            category = requested_category
            game_mode = requested_game or (FruitGames.MATCH if category == "interactive" else "")
            controller = InteractionController(model.default_scale, model.default_rotation)
            controller.state.input_mode = old_mode
            controller.state.debug = old_debug
            robot = RobotController(model) if category == "robotics" and model.loaded else None
            fruits = FruitGames(model, game_mode) if category == "interactive" and model.loaded else None
            ui.set_active_model(category)
            recognizer.reset()
            previous_input_mode = controller.state.input_mode
            processed_tracking_token = None
            raw_hit = None
            robot_drag_pointer = None

        def consume_command(command: UICommand | None) -> None:
            if command is None:
                return
            if command.kind == "tutorial_complete":
                _save_tutorial_state()
            elif command.kind == "select_model":
                activate(command.value)

        while not renderer.should_close():
            renderer.poll_events()
            _handle_keys(keys, renderer, ui, controller)
            if renderer.should_close():
                break

            now = time.perf_counter()
            delta_time = now - previous_time
            previous_time = now
            render_rate.tick(now)
            camera_frame = camera.latest()
            if camera_frame.sequence != submitted_camera_sequence:
                tracker.submit(camera_frame.image, camera_frame.captured_at)
                submitted_camera_sequence = camera_frame.sequence
            snapshot = _fresh_snapshot(tracker.latest(), now)
            if snapshot.sequence != previous_tracking_sequence:
                tracking_rate.tick(now)
                previous_tracking_sequence = snapshot.sequence

            state = controller.state
            if state.input_mode != previous_input_mode:
                recognizer.reset()
                gesture_frame = _empty_gesture(now)
                processed_tracking_token = None
                raw_hit = None
                previous_input_mode = state.input_mode
            tracking_token = (snapshot.sequence, snapshot.status, bool(snapshot.hands))
            has_new_hand_sample = state.input_mode == InputMode.HAND and tracking_token != processed_tracking_token
            scroll_delta = renderer.consume_scroll()

            if state.input_mode == InputMode.HAND and has_new_hand_sample:
                gesture_frame = recognizer.update(snapshot, now)
                ui_owned_before = ui.captures_input(gesture_frame.pointer, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
                command = ui.update(gesture_frame.pointer, gesture_frame.pinch_active, now, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
                ui_capture = bool(ui_owned_before or command is not None or ui.captures_input(gesture_frame.pointer, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
                consume_command(command)
                state = controller.state
                is_pointing = gesture_frame.pointer is not None and (gesture_frame.gesture in (Gesture.POINT, Gesture.PINCH) or gesture_frame.pinch_active) and not ui_capture and state.interaction != InteractionState.STOPPED
                raw_hit = renderer.pick(gesture_frame.pointer[0], gesture_frame.pointer[1], state.transform) if is_pointing else None
                pointed = None if raw_hit is None else raw_hit.component_id

                controller_frame = gesture_frame
                if (
                    fruits is not None
                    and state.interaction != InteractionState.STOPPED
                    and gesture_frame.gesture in (Gesture.OPEN_PALM, Gesture.V_SIGN, Gesture.TWO_HANDS)
                ):
                    controller_frame = replace(gesture_frame, gesture=Gesture.NONE, two_hand_distance=None)
                controller.update_hand(controller_frame, pointed, ui_capture=ui_capture, timestamp=now)

                if robot is not None:
                    if gesture_frame.gesture == Gesture.ROCK_SIGN:
                        rock_since = now if rock_since is None else rock_since
                        if not rock_latched and now - rock_since >= config.ROBOT_DEMO_HOLD_SECONDS:
                            robot.set_demo(not robot.demo_enabled)
                            rock_latched = True
                    else:
                        rock_since = None
                        rock_latched = False
                    selected_id = controller.state.selected_component
                    if gesture_frame.pinch_active and selected_id and gesture_frame.pointer is not None and not robot.demo_enabled:
                        if robot_drag_pointer is not None:
                            robot.add_drag(selected_id, gesture_frame.pointer[0] - robot_drag_pointer[0], gesture_frame.pointer[1] - robot_drag_pointer[1])
                        robot_drag_pointer = gesture_frame.pointer
                    else:
                        robot_drag_pointer = None
                if fruits is not None:
                    game_frame = gesture_frame if not ui_capture and state.interaction != InteractionState.STOPPED else _empty_gesture(now)
                    fruits.update(game_frame, controller.state.selected_component, controller.state.transform.scale)
                processed_tracking_token = tracking_token
            elif state.input_mode == InputMode.MOUSE:
                gesture_frame = _empty_gesture(now)
                pointer, left_down = renderer.mouse_state()
                ui_owned_before = ui.captures_input(pointer, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
                command = ui.update(pointer, False, now, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT, mouse_pressed=left_down)
                ui_capture = bool(ui_owned_before or command is not None or ui.captures_input(pointer, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
                consume_command(command)
                controller.update_mouse(pointer, left_down, scroll_delta, ui_capture=ui_capture)
                raw_hit = renderer.pick(pointer[0], pointer[1], controller.state.transform) if pointer and not ui_capture and not left_down else None
                controller.state.hovered_component = controller.hover.update(None if raw_hit is None else raw_hit.component_id, pointer, now)

            transform = controller.advance(delta_time)
            if robot is not None:
                robot.update(delta_time)
            state = controller.state
            display_selected_id = (
                robot.demo_joint
                if robot is not None and robot.demo_enabled and robot.demo_joint
                else state.selected_component
            )
            display_hovered_id = state.hovered_component
            hovered = model.component(display_hovered_id)
            selected = model.component(display_selected_id)
            active_id = display_selected_id or display_hovered_id
            callout_anchor = renderer.project_component(active_id, transform, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
            displayed_frame = camera_frame.image
            if state.debug:
                displayed_frame = camera_frame.image.copy()
                tracker.draw_debug(displayed_frame, snapshot.hands)

            error_title, error_detail = _visible_error(model, camera, tracker)
            jeremy.update_context(
                context_builder.build(
                    model=model,
                    model_path=config.MODEL_CATALOG[category][0],
                    category=category,
                    application_state=state,
                    selected_id=display_selected_id,
                    hovered_id=display_hovered_id,
                    active_gesture=gesture_frame.gesture,
                    tracking_status="MOUSE MODE" if state.input_mode == InputMode.MOUSE else snapshot.status,
                    camera_status=camera.status,
                    current_error="  /  ".join(part for part in (error_title, error_detail) if part),
                    game_mode=game_mode,
                    game_status="" if fruits is None else fruits.status,
                    robot=robot,
                )
            )
            jeremy_status = jeremy.status()
            game_boxes: tuple[tuple[str, float, float, float, float, bool, tuple[int, int, int]], ...] = ()
            if fruits is not None and game_mode == FruitGames.MATCH:
                matched_kinds = {fruits._kind(item) for item in fruits.matched}
                game_boxes = tuple((box.label, *box.rect, box.kind in matched_kinds, box.color_rgb) for box in fruits.boxes)
            show_education = category != "interactive"
            overlay = ui.draw(
                config.DISPLAY_WIDTH,
                config.DISPLAY_HEIGHT,
                HUDData(
                    tracking_status="MOUSE MODE" if state.input_mode == InputMode.MOUSE else snapshot.status,
                    category=category,
                    model_name=f"{model.category_name}  /  {model.model_name}",
                    instruction=_instruction(state.interaction, state.input_mode, gesture_frame.gesture, display_hovered_id, display_selected_id, ui.tutorial_visible, ui.models_open, gesture_frame.hand_count, category, game_mode),
                    pointer=state.pointer_position,
                    pinch_active=gesture_frame.pinch_active,
                    hovered_name=(
                        _anatomy_name(hovered)
                        if category == "anatomy"
                        else "" if hovered is None else hovered.name
                    ),
                    hovered_short=(
                        _anatomy_short(hovered)
                        if category == "anatomy"
                        else "" if hovered is None else hovered.short
                    ),
                    selected_name=(
                        _anatomy_name(selected)
                        if category == "anatomy" and show_education
                        else "" if selected is None or not show_education else selected.name
                    ),
                    selected_description=(
                        ""
                        if category == "anatomy" or selected is None or not show_education
                        else selected.description
                    ),
                    selected_fields=(
                        _component_fields(selected, robot, display_selected_id, category)
                        if show_education else ()
                    ),
                    callout_anchor=callout_anchor,
                    debug_lines=_debug_lines(render_rate=render_rate.displayed(now), tracking_rate=tracking_rate.displayed(now), snapshot=snapshot, gesture=gesture_frame, controller=controller, hit=raw_hit, raycast_ms=renderer.last_pick_ms, camera=camera, model=model) if state.debug else (),
                    stopped=state.interaction == InteractionState.STOPPED,
                    input_mode="MOUSE" if state.input_mode == InputMode.MOUSE else "HAND",
                    error_title=error_title,
                    error_detail=error_detail,
                    game_mode=game_mode,
                    game_status="" if fruits is None else fruits.status,
                    game_positive=False if fruits is None else fruits.status_positive,
                    game_boxes=game_boxes,
                    size_progress=0.0 if fruits is None else fruits.current_scale / fruits.target_scale,
                    jeremy_status=jeremy_status.label,
                    jeremy_detail=jeremy_status.detail,
                ),
            )
            renderer.render(displayed_frame, overlay, transform, hovered_component=display_hovered_id, selected_component=display_selected_id)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"Lumen could not start: {error}", file=sys.stderr)
        return 1
    finally:
        if jeremy is not None:
            jeremy.close()
        if tracker is not None:
            tracker.close()
        if camera is not None:
            camera.close()
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    raise SystemExit(run())
