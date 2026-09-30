"""Structured snapshots of the Lumen interface for Jeremy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from core.gestures import Gesture
from core.interaction import ApplicationState
from core.model import ComponentMetadata, SceneModel
from core.robotics import RobotController


@dataclass(frozen=True, slots=True)
class ComponentContext:
    id: str
    name: str
    short: str
    description: str
    fields: dict[str, str]


@dataclass(frozen=True, slots=True)
class TransformContext:
    position: tuple[float, float, float]
    rotation_degrees: tuple[float, float, float]
    zoom_scale: float


@dataclass(frozen=True, slots=True)
class LumenAIContext:
    current_model: str = ""
    current_model_file: str = ""
    current_model_category: str = ""
    selected_object: ComponentContext | None = None
    hovered_object: ComponentContext | None = None
    current_mode: str = ""
    input_mode: str = ""
    active_gesture: str = ""
    hand_tracking_status: str = ""
    camera_status: str = ""
    current_transform: TransformContext | None = None
    current_error: str = ""
    available_controls: tuple[str, ...] = ()
    game_mode: str = ""
    game_status: str = ""
    robot_demo_active: bool = False
    user_language: str = "English"

    def with_language(self, language: str) -> "LumenAIContext":
        return replace(self, user_language=language)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LumenContextBuilder:
    """Expose only state and controls that really exist in Lumen."""

    _BASE_HAND_CONTROLS = (
        "Point -> inspect or hover a component",
        "Pinch and hold -> select a component",
        "Pinch empty space -> clear the selection",
        "Open palm -> move the model",
        "V gesture -> rotate the model",
        "Two hands -> scale the model",
        "Fist -> stop interaction",
        "Open palm after stop -> resume interaction",
    )
    _MOUSE_CONTROLS = (
        "Left mouse drag -> rotate the model",
        "Mouse wheel -> scale the model",
    )

    @staticmethod
    def _component(
        model: SceneModel,
        component_id: str | None,
        robot: RobotController | None,
    ) -> ComponentContext | None:
        metadata: ComponentMetadata | None = model.component(component_id)
        if metadata is None or component_id is None:
            return None
        fields = dict(metadata.fields)
        if robot is not None:
            angle = robot.angle(component_id)
            limits = robot.limits(component_id)
            if angle is not None:
                fields["CURRENT ANGLE"] = f"{angle:.1f} degrees"
            if limits is not None:
                fields["JOINT LIMITS"] = f"{limits[0]:g} to {limits[1]:g} degrees"
            if component_id == "gripper":
                fields["OPEN"] = f"{robot.gripper_open * 100:.0f}%"
        return ComponentContext(
            id=metadata.id,
            name=metadata.name,
            short=metadata.short,
            description=metadata.description,
            fields=fields,
        )

    @classmethod
    def _controls(cls, category: str, input_mode: str) -> tuple[str, ...]:
        if input_mode.startswith("MOUSE"):
            controls = list(cls._MOUSE_CONTROLS)
        elif category == "interactive":
            controls = [
                "Point -> inspect or hover a fruit",
                "Pinch and hold -> select a fruit",
                "Pinch and drag a selected fruit -> move that fruit",
                "Open palm on a selected fruit -> move that fruit",
                "V gesture on a selected fruit -> rotate that fruit",
                "Two hands -> scale the selected fruit or size-game pear",
                "Fist -> stop interaction",
                "Open palm after stop -> resume interaction",
            ]
        else:
            controls = list(cls._BASE_HAND_CONTROLS)
        if category == "robotics":
            controls.extend(
                (
                    "Rock sign held -> start or stop the robot demonstration",
                    "Pinch and drag a selected joint -> move that joint or gripper",
                )
            )
        controls.extend(
            (
                "R key -> reset the view",
                "X key -> clear selection",
                "B key -> switch hand and mouse control",
            )
        )
        return tuple(controls)

    def build(
        self,
        *,
        model: SceneModel,
        model_path: str | Path,
        category: str,
        application_state: ApplicationState,
        selected_id: str | None,
        hovered_id: str | None,
        active_gesture: Gesture,
        tracking_status: str,
        camera_status: str,
        current_error: str = "",
        game_mode: str = "",
        game_status: str = "",
        robot: RobotController | None = None,
    ) -> LumenAIContext:
        transform = application_state.transform
        input_mode = application_state.input_mode.value
        return LumenAIContext(
            current_model=model.model_name,
            current_model_file=str(Path(model_path)),
            current_model_category=category,
            selected_object=self._component(model, selected_id, robot),
            hovered_object=self._component(model, hovered_id, robot),
            current_mode=application_state.interaction.value,
            input_mode=input_mode,
            active_gesture=active_gesture.value,
            hand_tracking_status=tracking_status,
            camera_status=camera_status,
            current_transform=TransformContext(
                position=tuple(map(float, transform.position)),
                rotation_degrees=tuple(map(float, transform.rotation)),
                zoom_scale=float(transform.scale),
            ),
            current_error=str(current_error or ""),
            available_controls=self._controls(category, input_mode),
            game_mode=game_mode,
            game_status=game_status,
            robot_demo_active=bool(robot is not None and robot.demo_enabled),
        )


__all__ = [
    "ComponentContext",
    "LumenAIContext",
    "LumenContextBuilder",
    "TransformContext",
]
