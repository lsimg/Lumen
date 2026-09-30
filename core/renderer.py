"""OpenGL camera compositor and material-preserving Lumen scene renderer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import glfw
import moderngl
import numpy as np
from PIL import Image

import config
from core.interaction import ModelTransform
from core.model import MeshPrimitive, RaycastHit, SceneModel


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    factor = 1.0 / math.tan(math.radians(fov_y) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = factor / max(aspect, 1e-6)
    matrix[1, 1] = factor
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def _translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    return matrix


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    return np.ascontiguousarray(matrix.T, dtype="f4").tobytes()


@dataclass(slots=True)
class _GpuPrimitive:
    source: MeshPrimitive
    vertex_buffer: moderngl.Buffer
    index_buffer: moderngl.Buffer
    vao: moderngl.VertexArray
    texture: moderngl.Texture

    def release(self) -> None:
        self.vao.release()
        self.vertex_buffer.release()
        self.index_buffer.release()


class Renderer:
    def __init__(self, width: int, height: int, model: SceneModel) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize. Update the graphics driver and try again.")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)
        self.window = glfw.create_window(width, height, config.WINDOW_TITLE, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create an OpenGL 3.3 window. Update the graphics driver.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if config.VSYNC else 0)
        glfw.set_window_aspect_ratio(self.window, width, height)

        self.ctx = moderngl.create_context(require=330)
        self.ctx.enable(moderngl.BLEND | moderngl.DEPTH_TEST)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.gpu_info = {key: self.ctx.info.get(key, "unknown") for key in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")}
        self.model = model
        self.last_pick_ms = 0.0
        self._last_pick_key: tuple[object, ...] | None = None
        self._last_pick_result: RaycastHit | None = None
        self._scroll_delta = 0.0
        glfw.set_scroll_callback(self.window, self._on_scroll)

        screen_vertex = self._read_shader("screen.vert")
        self._camera_program = self.ctx.program(vertex_shader=screen_vertex, fragment_shader=self._read_shader("camera.frag"))
        self._overlay_program = self.ctx.program(vertex_shader=screen_vertex, fragment_shader=self._read_shader("overlay.frag"))
        self._model_program = self.ctx.program(vertex_shader=self._read_shader("model.vert"), fragment_shader=self._read_shader("model.frag"))

        quad = np.asarray((-1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0), dtype=np.float32)
        self._quad_buffer = self.ctx.buffer(quad.tobytes())
        content = [(self._quad_buffer, "2f 2f", "in_position", "in_uv")]
        self._camera_vao = self.ctx.vertex_array(self._camera_program, content)
        self._overlay_vao = self.ctx.vertex_array(self._overlay_program, content)
        self._camera_texture = self.ctx.texture((width, height), 3, dtype="f1")
        self._overlay_texture = self.ctx.texture((width, height), 4, dtype="f1")
        self._camera_size = (width, height)
        self._overlay_size = (width, height)
        self._last_camera_frame: np.ndarray | None = None
        self._configure_screen_texture(self._camera_texture)
        self._configure_screen_texture(self._overlay_texture)
        self._camera_program["camera_texture"].value = 0
        self._camera_program["camera_texel_size"].value = (1.0 / width, 1.0 / height)
        self._camera_program["camera_sharpen"].value = config.CAMERA_SHARPEN_STRENGTH
        self._overlay_program["overlay_texture"].value = 1
        self._model_program["base_texture"].value = 2

        self._gpu_primitives: list[_GpuPrimitive] = []
        self._material_textures: dict[str, moderngl.Texture] = {}
        self._white_texture = self.ctx.texture((1, 1), 4, bytes((255, 255, 255, 255)))
        self._configure_model_texture(self._white_texture)
        self._highlight_levels = np.empty(0, dtype=np.float64)
        self._highlight_colors = np.empty((0, 3), dtype=np.float64)
        self._region_highlight_level = 0.0
        self._region_highlight_color = np.asarray(config.HOLOGRAM_COLOR, dtype=np.float64)
        self._highlight_time = time.perf_counter()
        self.set_model(model)

    @staticmethod
    def _read_shader(name: str) -> str:
        path = config.SHADER_DIR / name
        try:
            return path.read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError(f"Shader load error: {path.name}") from error

    @staticmethod
    def _configure_screen_texture(texture: moderngl.Texture) -> None:
        texture.filter = moderngl.LINEAR, moderngl.LINEAR
        texture.repeat_x = False
        texture.repeat_y = False

    @staticmethod
    def _configure_model_texture(texture: moderngl.Texture) -> None:
        texture.filter = moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR
        texture.repeat_x = True
        texture.repeat_y = True
        try:
            texture.build_mipmaps()
        except Exception:
            pass

    def _texture_for(self, image: Any | None, material_name: str) -> moderngl.Texture:
        if image is None:
            return self._white_texture
        key = str(material_name)
        cached = self._material_textures.get(key)
        if cached is not None:
            return cached
        try:
            pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
            rgba = pil.convert("RGBA")
            texture = self.ctx.texture(rgba.size, 4, rgba.tobytes())
            self._configure_model_texture(texture)
        except Exception:
            return self._white_texture
        self._material_textures[key] = texture
        return texture

    def _build_gpu_primitives(self, model: SceneModel) -> list[_GpuPrimitive]:
        result: list[_GpuPrimitive] = []
        try:
            for primitive in model.primitives:
                indices = primitive.line_indices if primitive.primitive_mode == "LINES" else primitive.triangle_indices
                if not len(primitive.vertices) or not len(indices):
                    continue
                interleaved = np.ascontiguousarray(np.column_stack((primitive.vertices, primitive.normals, primitive.uv)), dtype=np.float32)
                vertex_buffer = self.ctx.buffer(interleaved.tobytes())
                index_buffer = self.ctx.buffer(np.ascontiguousarray(indices, dtype=np.uint32).tobytes())
                vao = self.ctx.vertex_array(
                    self._model_program,
                    [(vertex_buffer, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
                    index_buffer=index_buffer,
                    index_element_size=4,
                )
                result.append(_GpuPrimitive(primitive, vertex_buffer, index_buffer, vao, self._texture_for(primitive.material.texture, primitive.material_name)))
        except Exception:
            for item in result:
                item.release()
            raise
        return result

    def set_model(self, model: SceneModel) -> None:
        for item in self._gpu_primitives:
            item.release()
        for texture in self._material_textures.values():
            texture.release()
        self._material_textures.clear()
        self.model = model
        self._gpu_primitives = self._build_gpu_primitives(model) if model.loaded else []
        count = len(self._gpu_primitives)
        self._highlight_levels = np.zeros(count, dtype=np.float64)
        self._highlight_colors = np.tile(np.asarray(config.HOLOGRAM_COLOR, dtype=np.float64), (count, 1))
        self._region_highlight_level = 0.0
        self._region_highlight_color = np.asarray(config.HOLOGRAM_COLOR, dtype=np.float64)
        self._last_pick_key = None
        self._last_pick_result = None

    def _on_scroll(self, _window: object, _x: float, y: float) -> None:
        self._scroll_delta += float(y)

    def consume_scroll(self) -> float:
        value, self._scroll_delta = self._scroll_delta, 0.0
        return value

    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.window))

    def request_close(self) -> None:
        glfw.set_window_should_close(self.window, True)

    def poll_events(self) -> None:
        glfw.poll_events()

    def key_down(self, key: int) -> bool:
        return glfw.get_key(self.window, key) == glfw.PRESS

    def mouse_state(self) -> tuple[tuple[float, float] | None, bool]:
        width, height = glfw.get_window_size(self.window)
        x, y = glfw.get_cursor_pos(self.window)
        pointer = ((float(x) / max(width, 1), float(y) / max(height, 1)) if 0.0 <= x <= width and 0.0 <= y <= height else None)
        down = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        if not glfw.get_window_attrib(self.window, glfw.FOCUSED):
            return None, False
        return pointer, bool(down)

    def _camera_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        width, height = glfw.get_framebuffer_size(self.window)
        return _perspective(config.MODEL_FOV_DEGREES, width / max(height, 1), 0.1, 100.0), _translation(0.0, 0.0, -config.MODEL_VIEW_DISTANCE)

    def pick(self, screen_x: float, screen_y: float, transform: ModelTransform) -> RaycastHit | None:
        key = (round(float(screen_x), 5), round(float(screen_y), 5), transform.position, transform.rotation, transform.scale, glfw.get_framebuffer_size(self.window), self.model.matrix_revision)
        if key == self._last_pick_key:
            self.last_pick_ms = 0.0
            return self._last_pick_result
        started = time.perf_counter()
        projection, view = self._camera_matrices()
        inverse = np.linalg.inv(projection @ view @ transform.matrix())
        near = inverse @ np.asarray((screen_x * 2.0 - 1.0, 1.0 - screen_y * 2.0, -1.0, 1.0))
        far = inverse @ np.asarray((screen_x * 2.0 - 1.0, 1.0 - screen_y * 2.0, 1.0, 1.0))
        near = near[:3] / near[3]
        far = far[:3] / far[3]
        result = self.model.raycast(near, far - near)
        self.last_pick_ms = (time.perf_counter() - started) * 1000.0
        self._last_pick_key = key
        self._last_pick_result = result
        return result

    def project_component(self, component_id: str | None, transform: ModelTransform, width: int, height: int) -> tuple[int, int] | None:
        center = self.model.target_center(component_id)
        if center is None:
            return None
        projection, view = self._camera_matrices()
        clip = projection @ view @ transform.matrix() @ np.append(center, 1.0)
        if clip[3] <= 0.0:
            return None
        ndc = clip[:2] / clip[3]
        if not np.isfinite(ndc).all() or np.any(np.abs(ndc) > 1.25):
            return None
        return int((ndc[0] + 1.0) * width * 0.5), int((1.0 - ndc[1]) * height * 0.5)

    def _write_camera(self, frame: np.ndarray) -> None:
        image = frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)
        height, width = image.shape[:2]
        if (width, height) != self._camera_size:
            self._camera_texture.release()
            self._camera_texture = self.ctx.texture((width, height), 3, dtype="f1")
            self._configure_screen_texture(self._camera_texture)
            self._camera_size = (width, height)
            self._camera_program["camera_texel_size"].value = (1.0 / width, 1.0 / height)
        self._camera_texture.write(image)

    def _write_overlay(self, overlay: np.ndarray) -> None:
        image = overlay if overlay.flags.c_contiguous else np.ascontiguousarray(overlay)
        height, width = image.shape[:2]
        if (width, height) != self._overlay_size:
            self._overlay_texture.release()
            self._overlay_texture = self.ctx.texture((width, height), 4, dtype="f1")
            self._configure_screen_texture(self._overlay_texture)
            self._overlay_size = (width, height)
        self._overlay_texture.write(image)

    def _target_style(self, target_id: str | None, hovered: str | None, selected: str | None) -> tuple[float, tuple[float, float, float]]:
        if target_id is not None and target_id == selected:
            return 0.92, config.SELECTED_COLOR
        if target_id is not None and target_id == hovered:
            return 0.68, config.HOVER_COLOR
        return 0.0, config.HOLOGRAM_COLOR

    def render(self, camera_frame: np.ndarray, overlay: np.ndarray, transform: ModelTransform, *, hovered_component: str | None, selected_component: str | None) -> None:
        if camera_frame is not self._last_camera_frame:
            self._write_camera(camera_frame)
            self._last_camera_frame = camera_frame
        self._write_overlay(overlay)
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        self.ctx.viewport = (0, 0, framebuffer_width, framebuffer_height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._camera_texture.use(location=0)
        self._camera_vao.render(moderngl.TRIANGLE_STRIP)

        projection, view = self._camera_matrices()
        root_matrix = transform.matrix()
        now = time.perf_counter()
        elapsed = now - self._highlight_time
        self._highlight_time = now
        blend = 1.0 - math.exp(-elapsed / max(config.HIGHLIGHT_RESPONSE_SECONDS, 1e-6))
        selected_regions = self.model.component_regions(selected_component)
        hovered_regions = self.model.component_regions(hovered_component)
        active_regions = selected_regions or hovered_regions
        desired_region_level = 0.92 if selected_regions else 0.68 if hovered_regions else 0.0
        desired_region_color = config.SELECTED_COLOR if selected_regions else config.HOVER_COLOR if hovered_regions else config.HOLOGRAM_COLOR
        self._region_highlight_level += (desired_region_level - self._region_highlight_level) * blend
        self._region_highlight_color += (np.asarray(desired_region_color) - self._region_highlight_color) * blend
        for index, gpu in enumerate(self._gpu_primitives):
            target = gpu.source.object_id or gpu.source.semantic_id
            level, color = self._target_style(target, hovered_component, selected_component)
            self._highlight_levels[index] += (level - self._highlight_levels[index]) * blend
            self._highlight_colors[index] += (np.asarray(color) - self._highlight_colors[index]) * blend

        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.screen.depth_mask = True
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        for index, gpu in enumerate(self._gpu_primitives):
            scene_matrix = self.model.node_matrix(gpu.source)
            primitive_matrix = root_matrix @ scene_matrix
            self._model_program["scene_matrix"].write(_matrix_bytes(scene_matrix))
            self._model_program["model_matrix"].write(_matrix_bytes(primitive_matrix))
            self._model_program["mvp"].write(_matrix_bytes(projection @ view @ primitive_matrix))
            self._model_program["base_color"].value = gpu.source.material.base_color
            self._model_program["emissive_color"].value = gpu.source.material.emissive
            self._model_program["metallic"].value = gpu.source.material.metallic
            self._model_program["roughness"].value = gpu.source.material.roughness
            self._model_program["highlight_color"].value = tuple(self._highlight_colors[index])
            self._model_program["highlight_level"].value = float(self._highlight_levels[index])
            self._model_program["model_brightness"].value = float(config.MODEL_BRIGHTNESS)
            self._model_program["opacity"].value = float(config.MODEL_OPACITY)
            self._model_program["scan_strength"].value = (
                0.015
                if max(self._highlight_levels[index], self._region_highlight_level) > 0.02
                else 0.0
            )
            shown_regions = active_regions[:4]
            self._model_program["region_count"].value = len(shown_regions)
            self._model_program["region_highlight_level"].value = float(self._region_highlight_level)
            self._model_program["region_highlight_color"].value = tuple(self._region_highlight_color)
            for region_index in range(4):
                if region_index < len(shown_regions):
                    min_x, min_y, min_z, max_x, max_y, max_z = shown_regions[region_index]
                    lower = (min_x, min_y, min_z)
                    upper = (max_x, max_y, max_z)
                else:
                    lower = (2.0, 2.0, 2.0)
                    upper = (2.0, 2.0, 2.0)
                self._model_program[f"region_min{region_index}"].value = lower
                self._model_program[f"region_max{region_index}"].value = upper
            gripper_spec = getattr(self.model, "robot_gripper_spec", {})
            is_gripper = bool(gripper_spec) and gpu.source.display_name == str(gripper_spec.get("node_label", ""))
            self._model_program["gripper_enabled"].value = 1.0 if is_gripper else 0.0
            self._model_program["gripper_open"].value = float(getattr(self.model, "robot_gripper_open", 1.0))
            self._model_program["gripper_params"].value = (
                float(gripper_spec.get("deform_start_x", -0.178852)),
                float(gripper_spec.get("deform_end_x", -0.27)),
                float(gripper_spec.get("hinge_y", 0.027204)),
                float(gripper_spec.get("hinge_z", 0.052931)),
            )
            self._model_program["gripper_closed_angle"].value = float(gripper_spec.get("closed_angle_degrees", 9.0))
            gpu.texture.use(location=2)
            if gpu.source.primitive_mode == "LINES":
                self.ctx.line_width = 2.0 if self._highlight_levels[index] > 0.2 else 1.0
                gpu.vao.render(moderngl.LINES)
            else:
                gpu.vao.render(moderngl.TRIANGLES)

        self.ctx.screen.depth_mask = True
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._overlay_texture.use(location=1)
        self._overlay_vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(self.window)

    def close(self) -> None:
        for item in getattr(self, "_gpu_primitives", ()):
            item.release()
        for texture in getattr(self, "_material_textures", {}).values():
            texture.release()
        for name in ("_white_texture", "_model_program", "_overlay_program", "_camera_program", "_overlay_vao", "_camera_vao", "_quad_buffer", "_overlay_texture", "_camera_texture"):
            resource = getattr(self, name, None)
            if resource is not None:
                resource.release()
        if getattr(self, "window", None):
            glfw.destroy_window(self.window)
        glfw.terminate()


__all__ = ["Renderer"]
