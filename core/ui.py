"""Compact OpenCV HUD rendered as a transparent BGRA overlay.

The renderer composites this overlay after drawing the camera and 3D model, so
interface controls remain visually above the scene as well as owning pointer
input while they are active.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import textwrap
from typing import Final, Sequence

import cv2
import numpy as np

from config import (
    PINCH_HOLD_SECONDS,
    UI_BACKGROUND_BGR,
    UI_BLUE_BGR,
    UI_CYAN_BGR,
    UI_MUTED_BGR,
    UI_PURPLE_BGR,
    UI_RED_BGR,
    UI_TEXT_BGR,
)


Point = tuple[float, float]
Rect = tuple[int, int, int, int]
Color = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class UICommand:
    """A small action emitted by an interactive HUD control."""

    kind: str
    value: str = ""


@dataclass(frozen=True, slots=True)
class HUDData:
    """All transient information required to draw one HUD frame."""

    tracking_status: str = ""
    category: str = ""
    model_name: str = ""
    instruction: str = ""
    pointer: Point | None = None
    pinch_active: bool = False
    hovered_name: str = ""
    hovered_short: str = ""
    selected_name: str = ""
    selected_description: str = ""
    selected_fields: Sequence[tuple[str, str]] = ()
    callout_anchor: Point | None = None
    debug_lines: Sequence[str] = ()
    stopped: bool = False
    input_mode: str = "HAND"
    error_title: str = ""
    error_detail: str = ""
    game_mode: str = ""
    game_status: str = ""
    game_positive: bool = False
    game_boxes: Sequence[tuple[str, float, float, float, float, bool, tuple[int, int, int]]] = ()
    size_progress: float = 0.0
    jeremy_status: str = ""
    jeremy_detail: str = ""


@dataclass(frozen=True, slots=True)
class _Item:
    item_id: str
    label: str
    action: str
    value: str = ""
    active: bool = False


@dataclass(frozen=True, slots=True)
class _SelectedCardLayout:
    box: Rect
    title_lines: tuple[str, ...]
    detail_lines: tuple[str, ...]
    field_lines: tuple[tuple[str, tuple[str, ...]], ...]


class LumenUI:
    """Minimal gesture-first HUD and its small amount of interaction state."""

    PANEL_WIDTH: Final[int] = 286
    CLOSED_WIDTH: Final[int] = 210
    OUTER_MARGIN: Final[int] = 14
    INNER_MARGIN: Final[int] = 10
    HEADER_HEIGHT: Final[int] = 58
    ITEM_HEIGHT: Final[int] = 34
    ITEM_GAP: Final[int] = 5

    _TUTORIAL_MAX_WIDTH: Final[int] = 530
    _TUTORIAL_MAX_HEIGHT: Final[int] = 400

    def __init__(self, tutorial_visible: bool) -> None:
        self.models_open = False
        self.active_model = "computer"
        self.tutorial_visible = bool(tutorial_visible)
        self.capture_latched = False
        self.mouse_capture_latched = False

        self._hovered_id: str | None = None
        self._candidate_id: str | None = None
        self._candidate_since = 0.0
        self._last_timestamp = 0.0
        self._pinch_was_active = False
        self._pinch_consumed = False
        self._mouse_was_pressed = False
        self._card_alpha = 0.0
        self._last_selected_data: HUDData | None = None

    def show_tutorial(self) -> None:
        """Open the tutorial from the HELP button or keyboard shortcut."""

        self.tutorial_visible = True
        self.models_open = False
        self._reset_candidate()

    def set_active_model(self, model_id: str) -> None:
        self.active_model = str(model_id)

    @staticmethod
    def _safe_text(value: object, limit: int = 48) -> str:
        text = (
            str(value or "")
            .replace("—", "-")
            .replace("→", ">")
            .replace("°", " DEG")
            .strip()
        )
        return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."

    @staticmethod
    def _point_pixels(
        pointer: Point | None,
        width: int,
        height: int,
        *,
        clamp: bool = False,
    ) -> tuple[int, int] | None:
        if pointer is None:
            return None
        try:
            x, y = float(pointer[0]), float(pointer[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not np.isfinite((x, y)).all():
            return None

        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            pixel_x = int(round(x * max(width - 1, 0)))
            pixel_y = int(round(y * max(height - 1, 0)))
        else:
            pixel_x, pixel_y = int(round(x)), int(round(y))
        if clamp and width > 0 and height > 0:
            pixel_x = int(np.clip(pixel_x, 0, width - 1))
            pixel_y = int(np.clip(pixel_y, 0, height - 1))
        return pixel_x, pixel_y

    def _items(self) -> tuple[_Item, ...]:
        if self.tutorial_visible:
            return (_Item("tutorial:start", "START EXPLORING", "tutorial_complete"),)
        if not self.models_open:
            return (
                _Item("models", "BACK / CATEGORIES", "open_models"),
                _Item("help", "HELP", "open_help"),
            )
        return (
            _Item("model:robotics", "ROBOTICS", "select_model", "robotics", self.active_model == "robotics"),
            _Item("model:computer", "COMPUTER", "select_model", "computer", self.active_model == "computer"),
            _Item("model:anatomy", "ANATOMY", "select_model", "anatomy", self.active_model == "anatomy"),
            _Item("model:interactive", "INTERACTIVE", "select_model", "interactive", self.active_model == "interactive"),
            *(
                (
                    _Item("game:match", "  MATCH THE FRUIT", "select_model", "interactive:match", False),
                    _Item("game:size", "  MATCH THE SIZE", "select_model", "interactive:size", False),
                )
                if self.active_model == "interactive" else ()
            ),
            _Item("close", "CLOSE", "close_models"),
        )

    def _layout(self, width: int, height: int) -> tuple[Rect, dict[str, Rect]]:
        items = self._items()
        if self.tutorial_visible:
            panel_width = min(self._TUTORIAL_MAX_WIDTH, max(width - 40, 1))
            panel_height = min(self._TUTORIAL_MAX_HEIGHT, max(height - 40, 1))
            left = max(0, (width - panel_width) // 2)
            top = max(0, (height - panel_height) // 2)
            button = (
                left + min(44, panel_width // 8),
                top + max(0, panel_height - 70),
                left + panel_width - min(44, panel_width // 8),
                top + max(1, panel_height - 30),
            )
            return (
                left,
                top,
                left + panel_width,
                top + panel_height,
            ), {items[0].item_id: button}

        panel_width = self.PANEL_WIDTH if self.models_open else self.CLOSED_WIDTH
        panel_width = min(panel_width, max(width - 2 * self.OUTER_MARGIN, 1))
        left = max(0, width - self.OUTER_MARGIN - panel_width)
        right = min(width, width - self.OUTER_MARGIN)
        top = min(self.OUTER_MARGIN, max(height - 1, 0))
        item_top = top + (self.HEADER_HEIGHT if self.models_open else 10)
        rectangles: dict[str, Rect] = {}
        for item in items:
            item_bottom = min(max(height - self.OUTER_MARGIN, item_top + 1), item_top + self.ITEM_HEIGHT)
            rectangles[item.item_id] = (
                min(left + self.INNER_MARGIN, right),
                item_top,
                max(left + self.INNER_MARGIN, right - self.INNER_MARGIN),
                item_bottom,
            )
            item_top = item_bottom + self.ITEM_GAP
        footer = 28 if self.models_open else 5
        panel_bottom = min(height, item_top + footer)
        return (left, top, right, panel_bottom), rectangles

    @staticmethod
    def _contains(rect: Rect, point: tuple[int, int]) -> bool:
        x, y = point
        return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]

    def _hit_test(
        self,
        pointer: Point | None,
        width: int,
        height: int,
    ) -> tuple[bool, str | None]:
        pixel = self._point_pixels(pointer, width, height)
        if pixel is None:
            return False, None
        panel, rectangles = self._layout(width, height)
        if not self._contains(panel, pixel):
            return False, None
        for item in self._items():
            rect = rectangles[item.item_id]
            if self._contains(rect, pixel):
                return True, item.item_id
        return True, None

    def captures_input(self, pointer: Point | None, width: int, height: int) -> bool:
        """Return whether the UI owns this pointer/pinch interaction."""

        if self.tutorial_visible:
            return True
        inside, _ = self._hit_test(pointer, width, height)
        return inside or self.capture_latched or self.mouse_capture_latched

    def _reset_candidate(self) -> None:
        self._candidate_id = None
        self._candidate_since = 0.0

    def _activate(self, item_id: str) -> UICommand | None:
        item = next((entry for entry in self._items() if entry.item_id == item_id), None)
        if item is None:
            return None
        if item.action == "open_models":
            self.models_open = True
        elif item.action == "close_models":
            self.models_open = False
        elif item.action == "open_help":
            self.show_tutorial()
        elif item.action == "tutorial_complete":
            self.tutorial_visible = False
            return UICommand("tutorial_complete")
        elif item.action == "select_model":
            self.models_open = False
            return UICommand("select_model", item.value)
        return None

    def update(
        self,
        pointer: Point | None,
        pinch_active: bool,
        timestamp: float,
        width: int,
        height: int,
        mouse_pressed: bool = False,
    ) -> UICommand | None:
        """Update UI hover/confirmation state and emit at most one command."""

        timestamp = max(float(timestamp), self._last_timestamp)
        self._last_timestamp = timestamp
        inside, hovered = self._hit_test(pointer, width, height)
        self._hovered_id = hovered

        mouse_pressed = bool(mouse_pressed)
        mouse_clicked = mouse_pressed and not self._mouse_was_pressed
        if mouse_pressed and inside:
            # Once a press touches the interface, own it until release even if
            # the cursor later leaves the panel.
            self.mouse_capture_latched = True
        elif not mouse_pressed:
            self.mouse_capture_latched = False
        self._mouse_was_pressed = mouse_pressed
        if mouse_clicked and hovered is not None:
            self._pinch_consumed = True
            self._reset_candidate()
            return self._activate(hovered)

        pinch_active = bool(pinch_active)
        pinch_started = pinch_active and not self._pinch_was_active
        self._pinch_was_active = pinch_active
        if not pinch_active:
            self.capture_latched = False
            self._pinch_consumed = False
            self._reset_candidate()
            return None

        if pinch_started and inside:
            self.capture_latched = True
        if not self.capture_latched or self._pinch_consumed or hovered is None:
            self._reset_candidate()
            return None

        if hovered != self._candidate_id:
            self._candidate_id = hovered
            self._candidate_since = timestamp
            return None
        if timestamp - self._candidate_since < PINCH_HOLD_SECONDS:
            return None

        self._pinch_consumed = True
        self._reset_candidate()
        return self._activate(hovered)

    def _hold_progress(self) -> float:
        if self._candidate_id is None or not self._pinch_was_active:
            return 0.0
        return float(
            np.clip(
                (self._last_timestamp - self._candidate_since)
                / max(PINCH_HOLD_SECONDS, 1e-6),
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _bgra(color: Color, alpha: int = 255) -> tuple[int, int, int, int]:
        return int(color[0]), int(color[1]), int(color[2]), int(alpha)

    @classmethod
    def _blend_rect(
        cls,
        overlay: np.ndarray,
        rect: Rect,
        color: Color,
        alpha: int,
    ) -> None:
        """Alpha-composite one solid rectangle into a straight-alpha overlay."""

        height, width = overlay.shape[:2]
        left = int(np.clip(rect[0], 0, width))
        top = int(np.clip(rect[1], 0, height))
        right = int(np.clip(rect[2], 0, width))
        bottom = int(np.clip(rect[3], 0, height))
        if right <= left or bottom <= top or alpha <= 0:
            return

        region = overlay[top:bottom, left:right]
        source_alpha = float(np.clip(alpha, 0, 255)) / 255.0
        destination_alpha = region[..., 3:4].astype(np.float32) / 255.0
        output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
        destination_rgb = region[..., :3].astype(np.float32)
        source_rgb = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        numerator = (
            source_rgb * source_alpha
            + destination_rgb * destination_alpha * (1.0 - source_alpha)
        )
        output_rgb = np.divide(
            numerator,
            output_alpha,
            out=np.zeros_like(numerator),
            where=output_alpha > 1e-6,
        )
        region[..., :3] = np.clip(output_rgb, 0.0, 255.0).astype(np.uint8)
        region[..., 3] = np.clip(output_alpha[..., 0] * 255.0, 0.0, 255.0).astype(np.uint8)

    @classmethod
    def _put_text(
        cls,
        overlay: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: Color = UI_TEXT_BGR,
        thickness: int = 1,
        alpha: int = 255,
    ) -> None:
        cv2.putText(
            overlay,
            cls._safe_text(text, 80),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            cls._bgra(color, alpha),
            thickness,
            cv2.LINE_AA,
        )

    @classmethod
    def _wrap_text_pixels(
        cls,
        text: str,
        max_width: int,
        scale: float,
        *,
        thickness: int = 1,
        max_lines: int | None = None,
    ) -> list[str]:
        """Wrap UI copy by rendered pixels, not an unreliable character count."""
        value = cls._safe_text(text, 500).strip()
        if not value or max_width <= 0:
            return []

        def fits(candidate: str) -> bool:
            return cv2.getTextSize(
                candidate,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                thickness,
            )[0][0] <= max_width

        words = value.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if fits(candidate):
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            # Mesh identifiers and similar labels may contain no spaces.
            while word and not fits(word):
                cut = len(word)
                while cut > 1 and not fits(word[:cut]):
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            current = word
        if current:
            lines.append(current)

        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1].rstrip(" .")
            while last and not fits(last + "..."):
                last = last[:-1].rstrip()
            lines[-1] = (last + "...") if last else "..."
        return lines

    def _selected_card_layout(
        self,
        width: int,
        height: int,
        data: HUDData,
    ) -> _SelectedCardLayout:
        """Measure a selected card before drawing so every baseline stays inside."""

        box_width = min(max(width - 28, 1), min(326, max(230, width // 3)))
        inner_width = max(1, box_width - 28)
        title_lines = self._wrap_text_pixels(
            data.selected_name.upper(), inner_width, 0.57, max_lines=4
        )
        detail_lines = self._wrap_text_pixels(
            data.selected_description, inner_width, 0.40, max_lines=3
        )
        field_lines: list[tuple[str, list[str]]] = []
        anatomy_allowed = {"SYSTEM", "LOCATION", "FUNCTION"}
        blocked_anatomy_text = (
            "unmapped",
            "unknown",
            "requires manual mapping",
            "object_",
            "mesh_",
            "material",
            "null",
            "n/a",
        )
        for heading, value in data.selected_fields:
            clean_heading = self._safe_text(heading, 18).upper()
            clean_value = str(value or "").strip()
            if data.category == "anatomy":
                if clean_heading not in anatomy_allowed:
                    continue
                lowered = clean_value.casefold()
                if any(marker in lowered for marker in blocked_anatomy_text):
                    continue
            if not clean_value:
                continue
            wrapped = self._wrap_text_pixels(clean_value, inner_width, 0.40, max_lines=3)
            if wrapped:
                field_lines.append((clean_heading, wrapped))

        def content_height() -> int:
            cursor = 27 + max(1, len(title_lines)) * 22
            if detail_lines:
                cursor += 2 + len(detail_lines) * 19
            for _heading, values in field_lines:
                cursor += 8 + 19 + len(values) * 19 + 4
            return cursor + 10

        preferred_top = 86 if height >= 300 else 14
        bottom_margin = 58 if height >= 300 else 14
        available_height = max(72, height - preferred_top - bottom_margin)

        # Extremely small windows still receive a valid compact card. Remove
        # optional copy before essential anatomy name/function information.
        while content_height() > available_height:
            multiline = next((values for _heading, values in reversed(field_lines) if len(values) > 1), None)
            if multiline is not None:
                multiline.pop()
                continue
            if detail_lines:
                detail_lines.pop()
                continue
            optional_index = next(
                (index for index, (heading, _values) in enumerate(field_lines) if heading == "LOCATION"),
                None,
            )
            if optional_index is not None:
                field_lines.pop(optional_index)
                continue
            if len(title_lines) > 1:
                title_lines.pop()
                continue
            break

        box_height = min(content_height(), max(72, height - 28))
        latest_y = max(preferred_top, height - box_height - bottom_margin)
        box_y = max(preferred_top, min(178, latest_y))
        box_x = max(14, width - box_width - 14)
        return _SelectedCardLayout(
            (box_x, box_y, box_x + box_width, box_y + box_height),
            tuple(title_lines),
            tuple(detail_lines),
            tuple((heading, tuple(values)) for heading, values in field_lines),
        )

    def _draw_hud(self, overlay: np.ndarray, data: HUDData) -> int:
        height, width = overlay.shape[:2]
        status = self._safe_text(data.tracking_status or "UNKNOWN", 18).upper()
        input_mode = self._safe_text(data.input_mode, 12).upper()
        status_line = f"TRACKING  {status}"
        if input_mode:
            status_line += f"  /  {input_mode}"
        lines = ["L U M E N", status_line]
        if data.model_name:
            lines.append(self._safe_text(data.model_name, 34).upper())
        if data.jeremy_status:
            jeremy = f"JEREMY  {self._safe_text(data.jeremy_status, 22).upper()}"
            if data.jeremy_detail:
                jeremy += f"  /  {self._safe_text(data.jeremy_detail, 34).upper()}"
            lines.append(jeremy)

        panel_width = min(max(width - 28, 1), 330)
        line_step = 23
        panel_height = 24 + len(lines) * line_step
        panel = (14, 14, 14 + panel_width, min(height, 14 + panel_height))
        self._blend_rect(overlay, panel, UI_BACKGROUND_BGR, 168)

        inactive_statuses = {
            "DISABLED",
            "LOST",
            "MODEL_MISSING",
            "NO HAND",
            "NO HANDS",
            "NO_HANDS",
            "OFF",
            "STALE",
            "STARTING",
            "TRACK LOST",
            "UNAVAILABLE",
        }
        tracking_color = UI_MUTED_BGR if status in inactive_statuses else UI_CYAN_BGR
        for index, text in enumerate(lines):
            if text.startswith("JEREMY"):
                color = (
                    UI_RED_BGR
                    if any(marker in text for marker in ("ERROR", "OFFLINE", "UNAVAILABLE", "CONFIG"))
                    else UI_PURPLE_BGR
                )
            else:
                color = tracking_color if index == 1 else UI_TEXT_BGR
            self._put_text(overlay, text, (28, 42 + index * line_step), 0.48, color)
        return panel[3]

    def _draw_debug(self, overlay: np.ndarray, data: HUDData, top: int) -> None:
        lines = tuple(self._safe_text(line, 72) for line in data.debug_lines if str(line).strip())
        if not lines:
            return
        height, width = overlay.shape[:2]
        line_step = 17
        available = max(0, (height - top - 28) // line_step)
        lines = lines[:available]
        if not lines:
            return
        text_width = max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
            for line in lines
        )
        panel_width = min(width - 28, max(250, text_width + 24))
        panel = (14, top + 6, 14 + panel_width, top + 20 + len(lines) * line_step)
        self._blend_rect(overlay, panel, UI_BACKGROUND_BGR, 190)
        cv2.rectangle(
            overlay,
            panel[:2],
            (panel[2] - 1, panel[3] - 1),
            self._bgra(UI_CYAN_BGR),
            1,
            cv2.LINE_AA,
        )
        for index, line in enumerate(lines):
            self._put_text(
                overlay,
                line,
                (panel[0] + 10, panel[1] + 17 + index * line_step),
                0.36,
                UI_TEXT_BGR,
            )

    def _draw_instruction(self, overlay: np.ndarray, instruction: str) -> None:
        if not instruction:
            return
        height, width = overlay.shape[:2]
        text = self._safe_text(instruction, 60).upper()
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.53, 1)[0][0]
        x = max(14, (width - text_width) // 2)
        self._put_text(overlay, text, (x, height - 24), 0.53, UI_CYAN_BGR)

    def _draw_callout(self, overlay: np.ndarray, data: HUDData) -> None:
        selected = bool(data.selected_name)
        name = data.selected_name if selected else data.hovered_name
        if not name:
            return
        height, width = overlay.shape[:2]
        color = UI_PURPLE_BGR if selected else UI_BLUE_BGR
        if selected:
            # The selected educational card is intentionally anchored to the
            # right edge: it is larger than hover text but leaves the model in
            # view and keeps the original compact Lumen visual language.
            layout = self._selected_card_layout(width, height, data)
            box = layout.box
            box_x, box_y = box[:2]
            opacity = int(np.clip(self._card_alpha, 0.0, 1.0) * 255)
            self._blend_rect(overlay, box, UI_BACKGROUND_BGR, int(218 * self._card_alpha))
            cv2.rectangle(overlay, box[:2], (box[2] - 1, box[3] - 1), self._bgra(color, opacity), 1, cv2.LINE_AA)
            y = box_y + 27
            for line in layout.title_lines:
                self._put_text(overlay, line, (box_x + 14, y), 0.57, color, 1, opacity)
                y += 22
            if layout.detail_lines:
                y += 2
                for line in layout.detail_lines:
                    self._put_text(overlay, line, (box_x + 14, y), 0.40, UI_TEXT_BGR, 1, opacity)
                    y += 19
            for heading, values in layout.field_lines:
                y += 8
                self._put_text(overlay, heading, (box_x + 14, y), 0.34, UI_MUTED_BGR, 1, opacity)
                y += 19
                for value in values:
                    self._put_text(overlay, value, (box_x + 14, y), 0.40, UI_TEXT_BGR, 1, opacity)
                    y += 19
                y += 4
            return

        if data.callout_anchor is None:
            return
        anchor = self._point_pixels(data.callout_anchor, width, height, clamp=True)
        if anchor is None:
            return
        x, y = anchor
        box_width = min(300, max(190, width // 3))
        inner_width = box_width - 18
        name_lines = self._wrap_text_pixels(name.upper(), inner_width, 0.47, max_lines=3)
        detail_lines = self._wrap_text_pixels(data.hovered_short, inner_width, 0.39, max_lines=2)
        lines = [*(('name', line) for line in name_lines), *(('detail', line) for line in detail_lines)]
        line_height = 18
        box_height = 16 + len(lines) * line_height
        place_left = x > width * 0.62
        if place_left:
            box_x = max(12, x - box_width - 42)
        else:
            box_x = min(max(12, width - box_width - 12), x + 42)
        box_y = int(np.clip(y - box_height // 2, 50, max(50, height - box_height - 56)))
        box = (box_x, box_y, box_x + box_width, box_y + box_height)
        line_end = (box[2] if place_left else box[0], box_y + 16)

        cv2.circle(overlay, anchor, 3, self._bgra(color), 1, cv2.LINE_AA)
        cv2.line(overlay, anchor, line_end, self._bgra(color), 1, cv2.LINE_AA)
        self._blend_rect(overlay, box, UI_BACKGROUND_BGR, 190)
        cv2.line(
            overlay,
            (box[0], box[1]),
            (box[2], box[1]),
            self._bgra(color),
            1,
            cv2.LINE_AA,
        )
        for index, (kind, line) in enumerate(lines):
            self._put_text(
                overlay,
                line,
                (box_x + 9, box_y + 18 + index * line_height),
                0.47 if kind == "name" else 0.39,
                color if kind == "name" else UI_TEXT_BGR,
            )

    def _draw_game(self, overlay: np.ndarray, data: HUDData) -> None:
        if not data.game_mode:
            return
        height, width = overlay.shape[:2]
        if data.game_mode == "match":
            for label, left, top, right, bottom, matched, color_rgb in data.game_boxes:
                rect = (int(left * width), int(top * height), int(right * width), int(bottom * height))
                color = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                self._blend_rect(overlay, rect, color, 76 if matched else 45)
                cv2.rectangle(overlay, rect[:2], (rect[2] - 1, rect[3] - 1), self._bgra(color), 3 if matched else 2, cv2.LINE_AA)
                text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.51, 1)[0][0]
                text_x = rect[0] + max(8, (rect[2] - rect[0] - text_width) // 2)
                text_y = rect[1] + (rect[3] - rect[1]) // 2 + 6
                self._put_text(overlay, label, (text_x, text_y), 0.51, UI_TEXT_BGR)
        else:
            center = (int(width * 0.84), int(height * 0.48))
            axes = (int(width * 0.09), int(height * 0.16))
            cv2.ellipse(overlay, center, axes, 0, 0, 360, self._bgra(UI_BLUE_BGR), 2, cv2.LINE_AA)
            self._put_text(overlay, "PEAR OUTLINE", (center[0] - 62, center[1] - axes[1] - 14), 0.40, UI_BLUE_BGR)
            self._put_text(overlay, f"SIZE {data.size_progress:.0%}", (center[0] - 42, center[1] + axes[1] + 24), 0.42, UI_MUTED_BGR)
        if data.game_status:
            color = UI_CYAN_BGR if data.game_positive else UI_PURPLE_BGR
            self._put_text(overlay, data.game_status, (max(14, width // 2 - 100), height - 54), 0.55, color, 1)

    def _draw_items(
        self,
        overlay: np.ndarray,
        items: tuple[_Item, ...],
        rectangles: dict[str, Rect],
    ) -> None:
        progress = self._hold_progress()
        for item in items:
            rect = rectangles[item.item_id]
            hovered = item.item_id == self._hovered_id
            if item.active:
                self._blend_rect(overlay, rect, UI_PURPLE_BGR, 54)
            if hovered:
                self._blend_rect(overlay, rect, UI_BLUE_BGR, 61)
            color = UI_BLUE_BGR if hovered else UI_PURPLE_BGR if item.active else UI_CYAN_BGR
            cv2.rectangle(
                overlay,
                rect[:2],
                (rect[2] - 1, rect[3] - 1),
                self._bgra(color),
                2 if hovered else 1,
                cv2.LINE_AA,
            )
            self._put_text(
                overlay,
                item.label,
                (rect[0] + 9, rect[1] + 22),
                0.39,
                UI_PURPLE_BGR if item.active and not hovered else UI_TEXT_BGR,
            )
            if item.item_id == self._candidate_id and progress > 0.0:
                end_x = rect[0] + int((rect[2] - rect[0]) * progress)
                cv2.line(
                    overlay,
                    (rect[0], rect[3] - 2),
                    (end_x, rect[3] - 2),
                    self._bgra(UI_BLUE_BGR),
                    2,
                    cv2.LINE_AA,
                )

    def _draw_panel(self, overlay: np.ndarray) -> None:
        height, width = overlay.shape[:2]
        panel, rectangles = self._layout(width, height)
        if self.tutorial_visible:
            self._draw_tutorial(overlay, panel, rectangles)
            return

        self._blend_rect(
            overlay,
            panel,
            UI_BACKGROUND_BGR,
            194 if self.models_open else 173,
        )
        cv2.rectangle(
            overlay,
            panel[:2],
            (panel[2] - 1, panel[3] - 1),
            self._bgra(UI_CYAN_BGR),
            1,
            cv2.LINE_AA,
        )
        if self.models_open:
            self._put_text(
                overlay,
                "MODEL LIBRARY",
                (panel[0] + 12, panel[1] + 23),
                0.55,
                UI_TEXT_BGR,
            )
            self._put_text(
                overlay,
                "EDUCATION PATHS",
                (panel[0] + 12, panel[1] + 45),
                0.35,
                UI_MUTED_BGR,
            )
        self._draw_items(overlay, self._items(), rectangles)
        if self.models_open:
            self._put_text(
                overlay,
                "POINT + HOLD PINCH",
                (panel[0] + 12, panel[3] - 12),
                0.34,
                UI_MUTED_BGR,
            )

    def _draw_tutorial(
        self,
        overlay: np.ndarray,
        panel: Rect,
        rectangles: dict[str, Rect],
    ) -> None:
        self._blend_rect(overlay, panel, UI_BACKGROUND_BGR, 222)
        cv2.rectangle(
            overlay,
            panel[:2],
            (panel[2] - 1, panel[3] - 1),
            self._bgra(UI_CYAN_BGR),
            1,
            cv2.LINE_AA,
        )
        left = panel[0] + 35
        rows = (
            ("WELCOME TO LUMEN", "Explore four subjects with the same gestures.", UI_TEXT_BGR),
            ("POINT", "Identify a component.", UI_BLUE_BGR),
            ("PINCH + HOLD", "Show a longer explanation.", UI_PURPLE_BGR),
            ("OPEN PALM", "Move the model.", UI_BLUE_BGR),
            ("V", "Rotate the model.", UI_BLUE_BGR),
            ("TWO HANDS", "Scale the model.", UI_BLUE_BGR),
        )
        y = panel[1] + 37
        for index, (heading, detail, color) in enumerate(rows):
            if index == 0:
                self._put_text(overlay, heading, (left, y), 0.68, color)
                self._put_text(
                    overlay,
                    detail,
                    (left, y + 27),
                    0.38,
                    UI_MUTED_BGR,
                )
                y += 62
                continue
            self._put_text(overlay, heading, (left, y), 0.48, color)
            self._put_text(
                overlay,
                detail,
                (left + 150, y),
                0.38,
                UI_TEXT_BGR,
            )
            y += 42
        self._draw_items(overlay, self._items(), rectangles)

    def _draw_error(self, overlay: np.ndarray, data: HUDData) -> None:
        if not data.error_title and not data.error_detail:
            return
        height, width = overlay.shape[:2]
        detail_lines = textwrap.wrap(self._safe_text(data.error_detail, 120), width=38)[:2]
        card_width = min(330, max(width - 28, 1))
        card_height = 48 + len(detail_lines) * 18
        card = (14, max(14, height - card_height - 14), 14 + card_width, height - 14)
        self._blend_rect(overlay, card, UI_BACKGROUND_BGR, 218)
        cv2.rectangle(
            overlay,
            card[:2],
            (card[2] - 1, card[3] - 1),
            self._bgra(UI_RED_BGR),
            1,
            cv2.LINE_AA,
        )
        self._put_text(
            overlay,
            data.error_title or "LUMEN ERROR",
            (card[0] + 10, card[1] + 22),
            0.43,
            UI_RED_BGR,
        )
        for index, line in enumerate(detail_lines):
            self._put_text(
                overlay,
                line,
                (card[0] + 10, card[1] + 43 + index * 18),
                0.35,
                UI_TEXT_BGR,
            )

    def _draw_pointer(self, overlay: np.ndarray, data: HUDData) -> None:
        height, width = overlay.shape[:2]
        pointer = self._point_pixels(data.pointer, width, height)
        if pointer is None or not (0 <= pointer[0] < width and 0 <= pointer[1] < height):
            return
        cv2.circle(
            overlay,
            pointer,
            8 if data.pinch_active else 5,
            self._bgra(UI_CYAN_BGR),
            1,
            cv2.LINE_AA,
        )

    def draw(self, width: int, height: int, data: HUDData) -> np.ndarray:
        """Return a transparent BGRA HUD ready for post-scene compositing."""

        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            raise ValueError("HUD dimensions must be positive")
        overlay = np.zeros((height, width, 4), dtype=np.uint8)

        callout_data = data
        if data.selected_name:
            if self._last_selected_data is None or self._last_selected_data.selected_name != data.selected_name:
                self._card_alpha = 0.0
            self._last_selected_data = data
            self._card_alpha += (1.0 - self._card_alpha) * 0.18
        else:
            self._card_alpha *= 0.78
            if self._card_alpha > 0.02 and self._last_selected_data is not None:
                callout_data = replace(self._last_selected_data, hovered_name="", hovered_short="")
            elif self._card_alpha <= 0.02:
                self._last_selected_data = None
                self._card_alpha = 0.0

        hud_bottom = self._draw_hud(overlay, data)
        self._draw_debug(overlay, data, hud_bottom)
        if not self.tutorial_visible:
            self._draw_game(overlay, data)
            self._draw_callout(overlay, callout_data)
            self._draw_instruction(overlay, data.instruction)
        self._draw_panel(overlay)
        self._draw_error(overlay, data)
        self._draw_pointer(overlay, data)

        if data.stopped:
            cv2.rectangle(
                overlay,
                (3, 3),
                (width - 4, height - 4),
                self._bgra(UI_RED_BGR),
                4,
                cv2.LINE_AA,
            )
        return overlay


__all__ = ["HUDData", "LumenUI", "UICommand"]
