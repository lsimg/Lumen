"""Universal, material-preserving scene loading and CPU picking.

The loader keeps every authored primitive separate, preserves the node graph and
rest transforms, and caches all expensive render/raycast data once.  Educational
labels live in JSON metadata; unknown geometry is never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
import trimesh


SUPPORTED_MODEL_FORMATS = frozenset({".glb", ".gltf", ".stl", ".obj", ".ply", ".3mf"})


@dataclass(frozen=True, slots=True)
class ComponentMetadata:
    id: str
    name: str
    short: str
    description: str
    important: bool = False
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MaterialData:
    name: str
    base_color: tuple[float, float, float, float]
    metallic: float = 0.0
    roughness: float = 0.75
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    texture: Any | None = None


@dataclass(frozen=True, slots=True)
class SceneNode:
    name: str
    parent: str | None
    local_matrix: np.ndarray
    world_matrix: np.ndarray
    children: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MeshPrimitive:
    """One authored renderable primitive; it is never merged with neighbours."""

    node_name: str
    display_name: str
    mesh_name: str
    material_name: str
    material: MaterialData
    primitive_mode: str
    semantic_id: str | None
    object_id: str | None
    rest_matrix: np.ndarray
    inverse_rest_matrix: np.ndarray
    vertices: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    triangle_indices: np.ndarray
    line_indices: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    triangle_v0: np.ndarray
    triangle_edge1: np.ndarray
    triangle_edge2: np.ndarray
    triangle_chunk_bounds_min: np.ndarray
    triangle_chunk_bounds_max: np.ndarray
    triangle_chunk_size: int


# Compatibility name retained for current renderer imports while it is upgraded.
MeshGroup = MeshPrimitive


@dataclass(frozen=True, slots=True)
class RaycastHit:
    component_id: str | None
    distance: float
    position: tuple[float, float, float]
    node_name: str = ""
    mesh_name: str = ""
    material_name: str = ""
    object_id: str | None = None


@dataclass(frozen=True, slots=True)
class _Definition:
    id: str
    nodes: frozenset[str]
    meshes: frozenset[str]
    ancestors: frozenset[str]
    materials: frozenset[str]
    hotspots: tuple[tuple[float, float, float, float, float], ...] = ()
    regions: tuple[tuple[float, float, float, float, float, float], ...] = ()

    def matches(
        self,
        node_name: str,
        mesh_name: str,
        ancestor_names: frozenset[str],
        material_name: str,
    ) -> bool:
        return (
            node_name in self.nodes
            or mesh_name in self.meshes
            or bool(self.ancestors.intersection(ancestor_names))
            or material_name in self.materials
        )


class SceneModel:
    """Normalized scene shared by computer, robotics, anatomy, and games."""

    def __init__(
        self,
        *,
        primitives: tuple[MeshPrimitive, ...] = (),
        nodes: dict[str, SceneNode] | None = None,
        components: dict[str, ComponentMetadata] | None = None,
        target_centers: dict[str, tuple[float, float, float]] | None = None,
        hotspots: tuple[tuple[str, float, float, float, float], ...] = (),
        regions: tuple[tuple[str, float, float, float, float, float, float], ...] = (),
        model_name: str = "MODEL",
        category_name: str = "EDUCATION",
        default_scale: float = 0.82,
        default_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
        warning: str | None = None,
        loaded: bool = False,
    ) -> None:
        self.groups = tuple(primitives)  # old public name, intentionally retained
        self.primitives = self.groups
        self.nodes = dict(nodes or {})
        self.components = dict(components or {})
        self.model_name = model_name
        self.category_name = category_name
        self.default_scale = float(default_scale)
        self.default_rotation = tuple(float(value) for value in default_rotation)
        self.metadata = dict(metadata or {})
        self.error = error
        self.warning = warning
        self.loaded = bool(loaded)
        self._target_centers = dict(target_centers or {})
        # Some GLB exporters combine many independent motherboard headers into
        # a single mesh.  The official board-layout diagram supplies exact
        # board-space regions for those parts, allowing picking without
        # splitting, recolouring, or otherwise altering the authored model.
        self._hotspots = tuple(
            sorted(
                hotspots,
                key=lambda item: (item[3] - item[1]) * (item[4] - item[2]),
            )
        )
        self._regions = tuple(
            sorted(
                regions,
                key=lambda item: (item[4] - item[1]) * (item[5] - item[2]) * (item[6] - item[3]),
            )
        )
        region_members: dict[str, list[tuple[float, float, float, float, float, float]]] = {}
        for component_id, min_x, min_y, min_z, max_x, max_y, max_z in self._regions:
            region_members.setdefault(component_id, []).append((min_x, min_y, min_z, max_x, max_y, max_z))
        self._component_regions = {key: tuple(value) for key, value in region_members.items()}
        self._target_primitives: dict[str, tuple[MeshPrimitive, ...]] = {}
        target_members: dict[str, list[MeshPrimitive]] = {}
        for primitive in self.primitives:
            for target in (primitive.semantic_id, primitive.object_id):
                if target:
                    target_members.setdefault(target, []).append(primitive)
        self._target_primitives = {key: tuple(value) for key, value in target_members.items()}
        self._deformations: dict[str, np.ndarray] = {}
        self._object_deformations: dict[str, np.ndarray] = {}
        self.matrix_revision = 0

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> "SceneModel":
        source_path = Path(model_path)
        document: dict[str, Any] = {}
        warnings: list[str] = []
        components: dict[str, ComponentMetadata] = {}
        component_defs: tuple[_Definition, ...] = ()
        object_defs: tuple[_Definition, ...] = ()

        try:
            if metadata_path is not None:
                document = _read_metadata(Path(metadata_path))
                component_defs, components, component_warnings = _components(document)
                object_defs, object_warnings = _objects(document)
                warnings.extend(component_warnings)
                warnings.extend(object_warnings)
        except Exception as exc:
            warnings.append(f"METADATA LOAD ERROR\n{_safe_filename(metadata_path, 'metadata.json')}\n{_brief_error(exc)}")

        model_name = _clean_text(document.get("model_name"), source_path.stem.upper() or "MODEL")
        category_name = _clean_text(document.get("category_name"), "EDUCATION")
        default_scale = _positive_float(document.get("default_scale"), 0.82)
        default_rotation = _rotation(document.get("default_rotation"), (0.0, 0.0, 0.0))

        try:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if source_path.suffix.lower() not in SUPPORTED_MODEL_FORMATS:
                raise ValueError(f"unsupported model format: {source_path.suffix or '(none)'}")

            loaded = trimesh.load(source_path, force="scene", process=False)
            if isinstance(loaded, trimesh.Trimesh):
                loaded = trimesh.Scene(loaded)
            if not isinstance(loaded, trimesh.Scene) or not loaded.graph.nodes_geometry:
                raise ValueError("model contains no renderable geometry")

            bounds = np.asarray(loaded.bounds, dtype=np.float64)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
                raise ValueError("model has invalid bounds")
            extent = float(np.max(bounds[1] - bounds[0]))
            if extent <= 1e-10:
                raise ValueError("model has zero size")
            normalization = np.eye(4, dtype=np.float64)
            normalization[:3, :3] *= 2.0 / extent
            normalization[:3, 3] = -bounds.mean(axis=0) * (2.0 / extent)

            nodes, ancestors = _scene_nodes(loaded, normalization)
            material_lookup = _gltf_material_lookup(source_path)
            primitives: list[MeshPrimitive] = []
            hit_components: set[str] = set()
            allow_unknown = bool(document.get("allow_unknown_selection", False))

            for source_node in loaded.graph.nodes_geometry:
                source_matrix, geometry_name = loaded.graph[source_node]
                geometry = loaded.geometry.get(geometry_name)
                node_name = str(source_node)
                mesh_name = str(geometry_name)
                material = _material_data(geometry, material_lookup.get(mesh_name))
                ancestor_names = ancestors.get(node_name, frozenset())
                semantic_id = _matching_id(component_defs, node_name, mesh_name, ancestor_names, material.name)
                object_id = _matching_id(object_defs, node_name, mesh_name, ancestor_names, material.name)
                if semantic_id:
                    hit_components.add(semantic_id)
                elif allow_unknown:
                    semantic_id = f"unknown:{node_name}"
                    if semantic_id not in components:
                        templates = document.get("unknown_material_metadata", {})
                        template = templates.get(material.name, {}) if isinstance(templates, dict) else {}
                        if not isinstance(template, dict):
                            template = {}
                        template_fields = template.get("fields", {})
                        if not isinstance(template_fields, dict):
                            template_fields = {}
                        components[semantic_id] = ComponentMetadata(
                            id=semantic_id,
                            name=_clean_text(template.get("name"), "UNKNOWN_COMPONENT"),
                            short=_clean_text(template.get("short"), "This part requires manual mapping."),
                            description=_clean_text(template.get("description"), "Use inspection mode to identify this authored model part."),
                            fields={
                                **{str(key).upper(): str(value) for key, value in template_fields.items()},
                                "NODE": node_name,
                                "MESH": mesh_name,
                                "MATERIAL": material.name or "—",
                            },
                        )

                primitive = _build_primitive(
                    geometry=geometry,
                    node_name=node_name,
                    display_name=_display_name(node_name, ancestor_names, nodes),
                    mesh_name=mesh_name,
                    material=material,
                    semantic_id=semantic_id,
                    object_id=object_id,
                    rest_matrix=normalization @ np.asarray(source_matrix, dtype=np.float64),
                )
                if primitive is not None:
                    primitives.append(primitive)

            if not primitives:
                raise ValueError("model contains no usable triangles or lines")
            for definition in component_defs:
                if definition.id not in hit_components and not definition.hotspots and not definition.regions:
                    warnings.append(f"No model geometry matched component: {definition.id}")

            centers = _target_centers(primitives)
            hotspots: list[tuple[str, float, float, float, float]] = []
            regions: list[tuple[str, float, float, float, float, float, float]] = []
            for definition in component_defs:
                for min_x, min_y, max_x, max_y, anchor_z in definition.hotspots:
                    hotspots.append((definition.id, min_x, min_y, max_x, max_y))
                    centers.setdefault(
                        definition.id,
                        ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, anchor_z),
                    )
                for min_x, min_y, min_z, max_x, max_y, max_z in definition.regions:
                    regions.append((definition.id, min_x, min_y, min_z, max_x, max_y, max_z))
                    centers.setdefault(
                        definition.id,
                        ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5),
                    )
            return cls(
                primitives=tuple(primitives),
                nodes=nodes,
                components=components,
                target_centers=centers,
                hotspots=tuple(hotspots),
                regions=tuple(regions),
                model_name=model_name,
                category_name=category_name,
                default_scale=default_scale,
                default_rotation=default_rotation,
                metadata=document,
                warning="\n".join(warnings) or None,
                loaded=True,
            )
        except Exception as exc:
            return cls(
                components=components,
                model_name=model_name,
                category_name=category_name,
                default_scale=default_scale,
                default_rotation=default_rotation,
                metadata=document,
                error=f"MODEL LOAD ERROR\n{source_path.name or 'model'}\n{_brief_error(exc)}",
                warning="\n".join(warnings) or None,
            )

    def component(self, component_id: str | None) -> ComponentMetadata | None:
        return self.components.get(component_id) if component_id else None

    def component_center(self, component_id: str | None) -> tuple[float, float, float] | None:
        return self.target_center(component_id)

    def component_regions(
        self, component_id: str | None
    ) -> tuple[tuple[float, float, float, float, float, float], ...]:
        return self._component_regions.get(component_id or "", ())

    def target_center(self, target_id: str | None) -> tuple[float, float, float] | None:
        center = self._target_centers.get(target_id) if target_id else None
        if center is None:
            return None
        if self._deformations and target_id in self._target_primitives:
            points: list[np.ndarray] = []
            for primitive in self._target_primitives[target_id]:
                points.append(trimesh.transform_points(_box_corners(primitive.bounds_min, primitive.bounds_max), self.node_matrix(primitive)))
            combined = np.vstack(points)
            return tuple(float(value) for value in (combined.min(axis=0) + combined.max(axis=0)) * 0.5)
        deformation = self._object_deformations.get(target_id or "")
        if deformation is None:
            return center
        transformed = trimesh.transform_points(np.asarray(center).reshape(1, 3), deformation)[0]
        return tuple(float(value) for value in transformed)

    def node_matrix(self, primitive: MeshPrimitive) -> np.ndarray:
        matrix = primitive.rest_matrix
        deform = self._deformations.get(primitive.node_name)
        if deform is not None:
            matrix = deform @ matrix
        object_deform = self._object_deformations.get(primitive.object_id or "")
        return matrix if object_deform is None else object_deform @ matrix

    def node_inverse_matrix(self, primitive: MeshPrimitive) -> np.ndarray:
        if primitive.node_name not in self._deformations and (primitive.object_id or "") not in self._object_deformations:
            return primitive.inverse_rest_matrix
        return np.linalg.inv(self.node_matrix(primitive))

    def set_node_deformations(self, transforms: dict[str, np.ndarray]) -> None:
        cleaned: dict[str, np.ndarray] = {}
        for name, value in transforms.items():
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                cleaned[str(name)] = matrix.copy()
        self._deformations = cleaned
        self.matrix_revision += 1

    def clear_node_deformations(self) -> None:
        if self._deformations:
            self._deformations.clear()
            self.matrix_revision += 1

    def set_object_transform(self, object_id: str, transform: np.ndarray) -> None:
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("object transform must be a finite 4x4 matrix")
        self._object_deformations[str(object_id)] = matrix.copy()
        self.matrix_revision += 1

    def clear_object_transforms(self) -> None:
        if self._object_deformations:
            self._object_deformations.clear()
            self.matrix_revision += 1

    def raycast(
        self,
        origin: np.ndarray | tuple[float, float, float],
        direction: np.ndarray | tuple[float, float, float],
    ) -> RaycastHit | None:
        try:
            ray_origin = np.asarray(origin, dtype=np.float64).reshape(3)
            ray_direction = np.asarray(direction, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            return None
        length = float(np.linalg.norm(ray_direction))
        if length <= 1e-12 or not np.isfinite(ray_origin).all() or not np.isfinite(ray_direction).all():
            return None
        ray_direction /= length

        nearest: tuple[MeshPrimitive, float, np.ndarray] | None = None
        for primitive in self.primitives:
            try:
                matrix = self.node_matrix(primitive)
                inverse = self.node_inverse_matrix(primitive)
            except np.linalg.LinAlgError:
                continue
            local_origin = trimesh.transform_points(ray_origin.reshape(1, 3), inverse)[0]
            local_direction = inverse[:3, :3] @ ray_direction
            local_length = float(np.linalg.norm(local_direction))
            if local_length <= 1e-12:
                continue
            local_direction /= local_length
            if primitive.primitive_mode == "LINES":
                local_distance = _line_ray_distance(primitive, local_origin, local_direction)
            else:
                local_distance = _triangle_ray_distance(primitive, local_origin, local_direction)
            if local_distance is None:
                continue
            local_hit = local_origin + local_direction * local_distance
            world_hit = trimesh.transform_points(local_hit.reshape(1, 3), matrix)[0]
            world_distance = float(np.dot(world_hit - ray_origin, ray_direction))
            if world_distance <= 1e-7:
                continue
            if nearest is None or world_distance < nearest[1]:
                nearest = primitive, world_distance, world_hit

        if nearest is None:
            return None
        primitive, distance, position = nearest
        target_id = primitive.object_id or primitive.semantic_id
        # Exact authored-node matches remain authoritative.  Hotspots only
        # disambiguate otherwise-unlabelled geometry (for example Object_70,
        # which contains AAFP, TPM, T_SENSOR and PANEL in one primitive).
        if target_id is None or str(target_id).startswith("unknown:"):
            hit_x, hit_y, hit_z = (float(value) for value in position)
            for component_id, min_x, min_y, min_z, max_x, max_y, max_z in self._regions:
                if min_x <= hit_x <= max_x and min_y <= hit_y <= max_y and min_z <= hit_z <= max_z:
                    target_id = component_id
                    break
        if target_id is None or str(target_id).startswith("unknown:"):
            hit_x, hit_y = float(position[0]), float(position[1])
            for component_id, min_x, min_y, max_x, max_y in self._hotspots:
                if min_x <= hit_x <= max_x and min_y <= hit_y <= max_y:
                    target_id = component_id
                    break
        return RaycastHit(
            component_id=target_id,
            distance=distance,
            position=tuple(float(value) for value in position),
            node_name=primitive.node_name,
            mesh_name=primitive.mesh_name,
            material_name=primitive.material_name,
            object_id=primitive.object_id,
        )


# Existing imports keep working; the class is now category-independent.
MotherboardModel = SceneModel


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata root must be an object")
    return value


def _components(document: dict[str, Any]) -> tuple[tuple[_Definition, ...], dict[str, ComponentMetadata], list[str]]:
    raw_items = document.get("components", [])
    if not isinstance(raw_items, list):
        raise ValueError("metadata components must be a list")
    definitions: list[_Definition] = []
    metadata: dict[str, ComponentMetadata] = {}
    warnings: list[str] = []
    for index, raw in enumerate(raw_items):
        try:
            if not isinstance(raw, dict):
                raise ValueError("expected an object")
            item_id = _required_text(raw, "id")
            if item_id in metadata:
                raise ValueError(f"duplicate component id: {item_id}")
            selectors = _definition(item_id, raw.get("selectors"))
            fields = raw.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError("fields must be an object")
            metadata[item_id] = ComponentMetadata(
                id=item_id,
                name=_required_text(raw, "name"),
                short=_required_text(raw, "short"),
                description=_required_text(raw, "description"),
                important=bool(raw.get("important", False)),
                fields={str(key).upper(): str(value) for key, value in fields.items()},
            )
            definitions.append(selectors)
        except (TypeError, ValueError) as exc:
            warnings.append(f"Ignored metadata component {index}: {exc}")
    return tuple(definitions), metadata, warnings


def _objects(document: dict[str, Any]) -> tuple[tuple[_Definition, ...], list[str]]:
    raw_items = document.get("objects", [])
    if not isinstance(raw_items, list):
        raise ValueError("metadata objects must be a list")
    definitions: list[_Definition] = []
    warnings: list[str] = []
    for index, raw in enumerate(raw_items):
        try:
            if not isinstance(raw, dict):
                raise ValueError("expected an object")
            definitions.append(_definition(_required_text(raw, "id"), raw.get("selectors")))
        except (TypeError, ValueError) as exc:
            warnings.append(f"Ignored metadata object {index}: {exc}")
    return tuple(definitions), warnings


def _definition(item_id: str, raw_selectors: Any) -> _Definition:
    if not isinstance(raw_selectors, dict):
        raise ValueError("selectors must be an object")
    values = [_selector_values(raw_selectors, key) for key in ("nodes", "meshes", "ancestors", "materials")]
    hotspots = _hotspot_values(raw_selectors.get("hotspots"))
    regions = _region_values(raw_selectors.get("regions"))
    if not any(values) and not hotspots and not regions:
        raise ValueError("at least one exact selector, hotspot, or region is required")
    return _Definition(item_id, *values, hotspots, regions)


def _hotspot_values(raw: Any) -> tuple[tuple[float, float, float, float, float], ...]:
    """Read official-layout rectangles in normalized model coordinates.

    Entries use ``[left, top, right, bottom]``-independent numeric bounds as
    ``[min_x, min_y, max_x, max_y]`` plus an optional callout anchor Z.
    """

    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("hotspots must be a list")
    result: list[tuple[float, float, float, float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"hotspot {index} must be an object")
        rect = item.get("rect")
        if not isinstance(rect, list) or len(rect) != 4:
            raise ValueError(f"hotspot {index} rect must contain four numbers")
        try:
            min_x, min_y, max_x, max_y = (float(value) for value in rect)
            anchor_z = float(item.get("anchor_z", 0.08))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"hotspot {index} contains a non-numeric value") from exc
        if not np.isfinite((min_x, min_y, max_x, max_y, anchor_z)).all():
            raise ValueError(f"hotspot {index} contains a non-finite value")
        if min_x >= max_x or min_y >= max_y:
            raise ValueError(f"hotspot {index} bounds are reversed or empty")
        result.append((min_x, min_y, max_x, max_y, anchor_z))
    return tuple(result)


def _region_values(raw: Any) -> tuple[tuple[float, float, float, float, float, float], ...]:
    """Read axis-aligned 3D selection regions in normalized scene space."""

    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("regions must be a list")
    result: list[tuple[float, float, float, float, float, float]] = []
    for index, item in enumerate(raw):
        values = item.get("box") if isinstance(item, dict) else None
        if not isinstance(values, list) or len(values) != 6:
            raise ValueError(f"region {index} box must contain six numbers")
        try:
            min_x, min_y, min_z, max_x, max_y, max_z = (float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"region {index} contains a non-numeric value") from exc
        if not np.isfinite((min_x, min_y, min_z, max_x, max_y, max_z)).all():
            raise ValueError(f"region {index} contains a non-finite value")
        if min_x >= max_x or min_y >= max_y or min_z >= max_z:
            raise ValueError(f"region {index} bounds are reversed or empty")
        result.append((min_x, min_y, min_z, max_x, max_y, max_z))
    return tuple(result)


def _matching_id(
    definitions: tuple[_Definition, ...],
    node_name: str,
    mesh_name: str,
    ancestors: frozenset[str],
    material_name: str,
) -> str | None:
    for definition in definitions:
        if definition.matches(node_name, mesh_name, ancestors, material_name):
            return definition.id
    return None


def _scene_nodes(scene: trimesh.Scene, normalization: np.ndarray) -> tuple[dict[str, SceneNode], dict[str, frozenset[str]]]:
    parents = scene.graph.transforms.parents
    base = scene.graph.base_frame
    children_map: dict[str, list[str]] = {}
    for child, parent in parents.items():
        children_map.setdefault(str(parent), []).append(str(child))
    nodes: dict[str, SceneNode] = {}
    ancestors: dict[str, frozenset[str]] = {}
    for raw_name in scene.graph.nodes:
        name = str(raw_name)
        world, _ = scene.graph[raw_name]
        parent_raw = parents.get(raw_name)
        parent = None if parent_raw is None or parent_raw == base else str(parent_raw)
        if parent_raw is None:
            local = np.asarray(world, dtype=np.float64)
        else:
            parent_world, _ = scene.graph[parent_raw]
            local = np.linalg.inv(np.asarray(parent_world, dtype=np.float64)) @ np.asarray(world, dtype=np.float64)
        chain = _ancestor_names(raw_name, parents, base)
        ancestors[name] = chain
        nodes[name] = SceneNode(
            name=name,
            parent=parent,
            local_matrix=_readonly(np.asarray(local, dtype=np.float64)),
            world_matrix=_readonly(normalization @ np.asarray(world, dtype=np.float64)),
            children=tuple(children_map.get(name, ())),
        )
    return nodes, ancestors


def _ancestor_names(source_node: Any, parents: Any, base: Any) -> frozenset[str]:
    names: set[str] = set()
    seen: set[Any] = set()
    cursor = source_node
    while cursor in parents and cursor not in seen:
        seen.add(cursor)
        parent = parents[cursor]
        if parent == base:
            break
        names.add(str(parent))
        cursor = parent
    return frozenset(names)


def _display_name(node_name: str, ancestors: frozenset[str], nodes: dict[str, SceneNode]) -> str:
    # Exporters often put useful labels on a transform parent and generic labels
    # on its mesh child. Prefer that parent without altering the hierarchy.
    node = nodes.get(node_name)
    if node and node.parent and node.parent in ancestors and len(nodes[node.parent].children) == 1:
        return node.parent
    return node_name


def _gltf_material_lookup(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() not in {".glb", ".gltf"}:
        return {}
    try:
        if path.suffix.lower() == ".glb":
            with path.open("rb") as stream:
                magic, version, _total_length = struct.unpack("<4sII", stream.read(12))
                if magic != b"glTF" or version != 2:
                    return {}
                chunk_length, chunk_type = struct.unpack("<II", stream.read(8))
                if chunk_type != 0x4E4F534A:
                    return {}
                tree = json.loads(stream.read(chunk_length).decode("utf-8").rstrip("\x00 \t\r\n"))
        else:
            tree = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(tree, dict):
            return {}
        meshes = tree.get("meshes", [])
        materials = tree.get("materials", [])
        result: dict[str, dict[str, Any]] = {}
        for mesh in meshes:
            name = str(mesh.get("name", ""))
            primitives = mesh.get("primitives", [])
            if name and primitives:
                index = primitives[0].get("material")
                if isinstance(index, int) and 0 <= index < len(materials):
                    result[name] = materials[index]
        return result
    except Exception:
        return {}


def _material_data(geometry: Any, raw: dict[str, Any] | None) -> MaterialData:
    visual = getattr(geometry, "visual", None)
    source = getattr(visual, "material", None)
    pbr = (raw or {}).get("pbrMetallicRoughness", {})
    name = str((raw or {}).get("name") or getattr(source, "name", "") or "MATERIAL")
    color = pbr.get("baseColorFactor", getattr(source, "baseColorFactor", None))
    if color is None:
        color = getattr(source, "diffuse", (204, 204, 204, 255))
    base = _color4(color)
    metallic = _unit_float(pbr.get("metallicFactor", getattr(source, "metallicFactor", 0.0)), 0.0)
    roughness = _unit_float(pbr.get("roughnessFactor", getattr(source, "roughnessFactor", 0.75)), 0.75)
    emissive = _color3((raw or {}).get("emissiveFactor", getattr(source, "emissiveFactor", (0, 0, 0))))
    texture = getattr(source, "baseColorTexture", None)
    if texture is None:
        texture = getattr(source, "image", None)
    return MaterialData(name, base, metallic, roughness, emissive, texture)


def _build_primitive(
    *,
    geometry: Any,
    node_name: str,
    display_name: str,
    mesh_name: str,
    material: MaterialData,
    semantic_id: str | None,
    object_id: str | None,
    rest_matrix: np.ndarray,
) -> MeshPrimitive | None:
    vertices: np.ndarray
    faces: np.ndarray
    mode: str
    if isinstance(geometry, trimesh.Trimesh):
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
        faces = np.asarray(geometry.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            return None
        mode = "TRIANGLES"
        cached_normals = getattr(getattr(geometry, "_cache", None), "cache", {}).get("vertex_normals")
        normals = np.asarray(cached_normals, dtype=np.float32) if cached_normals is not None else _vertex_normals(vertices, faces)
        uv = np.asarray(getattr(geometry.visual, "uv", np.empty((0, 2))), dtype=np.float32)
        if uv.shape != (len(vertices), 2):
            uv = np.zeros((len(vertices), 2), dtype=np.float32)
        triangles = vertices[faces]
        triangle_indices = faces.reshape(-1).astype(np.uint32)
        line_indices = np.empty(0, dtype=np.uint32)
        triangle_v0 = triangles[:, 0]
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        chunk_size = 64
        chunk_min, chunk_max = _triangle_chunk_bounds(triangles, chunk_size)
    elif hasattr(geometry, "entities") and hasattr(geometry, "vertices"):
        # Read Path3D line entities directly.  ``Path3D.discrete`` pulls in the
        # optional networkx package, while GL_LINES only needs authored point
        # order and therefore has no graph dependency.
        source_vertices = np.asarray(geometry.vertices, dtype=np.float64)
        polylines: list[np.ndarray] = []
        for entity in geometry.entities:
            point_ids = np.asarray(getattr(entity, "points", ()), dtype=np.int64)
            if len(point_ids) >= 2:
                polylines.append(source_vertices[point_ids])
        if not polylines:
            return None
        vertices = np.vstack(polylines)
        indices: list[int] = []
        offset = 0
        for line in polylines:
            for index in range(len(line) - 1):
                indices.extend((offset + index, offset + index + 1))
            offset += len(line)
        mode = "LINES"
        normals = np.zeros_like(vertices, dtype=np.float32)
        uv = np.zeros((len(vertices), 2), dtype=np.float32)
        triangle_indices = np.empty(0, dtype=np.uint32)
        line_indices = np.asarray(indices, dtype=np.uint32)
        triangle_v0 = np.empty((0, 3), dtype=np.float64)
        edge1 = np.empty((0, 3), dtype=np.float64)
        edge2 = np.empty((0, 3), dtype=np.float64)
        chunk_size = 0
        chunk_min = np.empty((0, 3), dtype=np.float64)
        chunk_max = np.empty((0, 3), dtype=np.float64)
    else:
        return None

    return MeshPrimitive(
        node_name=node_name,
        display_name=display_name,
        mesh_name=mesh_name,
        material_name=material.name,
        material=material,
        primitive_mode=mode,
        semantic_id=semantic_id,
        object_id=object_id,
        rest_matrix=_readonly(np.asarray(rest_matrix, dtype=np.float64)),
        inverse_rest_matrix=_readonly(np.linalg.inv(np.asarray(rest_matrix, dtype=np.float64))),
        vertices=_readonly(np.asarray(vertices, dtype=np.float32)),
        normals=_readonly(np.ascontiguousarray(normals, dtype=np.float32)),
        uv=_readonly(np.ascontiguousarray(uv, dtype=np.float32)),
        triangle_indices=_readonly(np.ascontiguousarray(triangle_indices, dtype=np.uint32)),
        line_indices=_readonly(np.ascontiguousarray(line_indices, dtype=np.uint32)),
        bounds_min=_readonly(vertices.min(axis=0).astype(np.float64)),
        bounds_max=_readonly(vertices.max(axis=0).astype(np.float64)),
        triangle_v0=_readonly(np.ascontiguousarray(triangle_v0, dtype=np.float64)),
        triangle_edge1=_readonly(np.ascontiguousarray(edge1, dtype=np.float64)),
        triangle_edge2=_readonly(np.ascontiguousarray(edge2, dtype=np.float64)),
        triangle_chunk_bounds_min=_readonly(np.ascontiguousarray(chunk_min, dtype=np.float64)),
        triangle_chunk_bounds_max=_readonly(np.ascontiguousarray(chunk_max, dtype=np.float64)),
        triangle_chunk_size=chunk_size,
    )


def _target_centers(primitives: list[MeshPrimitive]) -> dict[str, tuple[float, float, float]]:
    buckets: dict[str, list[np.ndarray]] = {}
    for primitive in primitives:
        for target in (primitive.semantic_id, primitive.object_id):
            if target:
                points = trimesh.transform_points(
                    np.vstack((primitive.bounds_min, primitive.bounds_max)), primitive.rest_matrix
                )
                buckets.setdefault(target, []).append(points)
    result: dict[str, tuple[float, float, float]] = {}
    for target, parts in buckets.items():
        points = np.vstack(parts)
        result[target] = tuple(float(value) for value in (points.min(axis=0) + points.max(axis=0)) * 0.5)
    return result


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    return np.asarray(normals, dtype=np.float32)


def _box_corners(bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    return np.asarray(
        [(x, y, z) for x in (bounds_min[0], bounds_max[0]) for y in (bounds_min[1], bounds_max[1]) for z in (bounds_min[2], bounds_max[2])],
        dtype=np.float64,
    )


def _triangle_ray_distance(primitive: MeshPrimitive, origin: np.ndarray, direction: np.ndarray) -> float | None:
    if not _ray_intersects_bounds(origin, direction, primitive.bounds_min, primitive.bounds_max):
        return None
    candidate_indices: np.ndarray | slice = slice(None)
    if len(primitive.triangle_chunk_bounds_min) > 1:
        hit_chunks = np.flatnonzero(
            _rays_intersect_bounds(
                origin,
                direction,
                primitive.triangle_chunk_bounds_min,
                primitive.triangle_chunk_bounds_max,
            )
        )
        if not len(hit_chunks):
            return None
        size = primitive.triangle_chunk_size
        count = len(primitive.triangle_v0)
        candidate_indices = np.concatenate(
            [np.arange(index * size, min((index + 1) * size, count), dtype=np.int64) for index in hit_chunks]
        )
    v0 = primitive.triangle_v0[candidate_indices]
    edge1 = primitive.triangle_edge1[candidate_indices]
    edge2 = primitive.triangle_edge2[candidate_indices]
    p = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, p)
    valid = np.abs(determinant) > 1e-9
    if not np.any(valid):
        return None
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    t = origin - v0
    u = np.einsum("ij,ij->i", t, p) * inverse
    valid &= (u >= 0.0) & (u <= 1.0)
    q = np.cross(t, edge1)
    v = np.einsum("j,ij->i", direction, q) * inverse
    valid &= (v >= 0.0) & ((u + v) <= 1.0)
    distance = np.einsum("ij,ij->i", edge2, q) * inverse
    valid &= distance > 1e-7
    return float(np.min(distance[valid])) if np.any(valid) else None


def _line_ray_distance(primitive: MeshPrimitive, origin: np.ndarray, direction: np.ndarray) -> float | None:
    if not len(primitive.line_indices):
        return None
    segments = primitive.vertices[primitive.line_indices.reshape(-1, 2)].astype(np.float64)
    start = segments[:, 0]
    edge = segments[:, 1] - start
    w0 = origin - start
    a = np.einsum("ij,ij->i", edge, edge)
    b = edge @ direction
    d = np.einsum("ij,ij->i", edge, w0)
    e = w0 @ direction
    denominator = np.maximum(a - b * b, 1e-12)
    segment_t = np.clip((b * e - d) / denominator, 0.0, 1.0)
    ray_t = np.maximum(b * segment_t - e, 0.0)
    closest_segment = start + edge * segment_t[:, None]
    closest_ray = origin + direction * ray_t[:, None]
    distance = np.linalg.norm(closest_segment - closest_ray, axis=1)
    valid = distance <= 0.012
    return float(np.min(ray_t[valid])) if np.any(valid) else None


def _triangle_chunk_bounds(triangles: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    minimums: list[np.ndarray] = []
    maximums: list[np.ndarray] = []
    for start in range(0, len(triangles), chunk_size):
        block = triangles[start : start + chunk_size]
        minimums.append(block.min(axis=(0, 1)))
        maximums.append(block.max(axis=(0, 1)))
    return np.asarray(minimums, dtype=np.float64), np.asarray(maximums, dtype=np.float64)


def _rays_intersect_bounds(origin: np.ndarray, direction: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    near = np.zeros(len(bounds_min), dtype=np.float64)
    far = np.full(len(bounds_min), np.inf, dtype=np.float64)
    valid = np.ones(len(bounds_min), dtype=bool)
    for axis in range(3):
        if abs(direction[axis]) <= 1e-12:
            valid &= (origin[axis] >= bounds_min[:, axis]) & (origin[axis] <= bounds_max[:, axis])
            continue
        first = (bounds_min[:, axis] - origin[axis]) / direction[axis]
        second = (bounds_max[:, axis] - origin[axis]) / direction[axis]
        near = np.maximum(near, np.minimum(first, second))
        far = np.minimum(far, np.maximum(first, second))
        valid &= far >= near
    return valid


def _ray_intersects_bounds(origin: np.ndarray, direction: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> bool:
    near, far = 0.0, np.inf
    for axis in range(3):
        if abs(direction[axis]) <= 1e-12:
            if origin[axis] < bounds_min[axis] or origin[axis] > bounds_max[axis]:
                return False
            continue
        first = (bounds_min[axis] - origin[axis]) / direction[axis]
        second = (bounds_max[axis] - origin[axis]) / direction[axis]
        near = max(near, min(first, second))
        far = min(far, max(first, second))
        if far < near:
            return False
    return True


def _selector_values(mapping: dict[str, Any], key: str) -> frozenset[str]:
    values = mapping.get(key, [])
    if not isinstance(values, list):
        raise ValueError(f"selector {key} must be a list")
    return frozenset(str(value).strip() for value in values if str(value).strip())


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _clean_text(value: Any, fallback: str) -> str:
    return str(value or "").strip() or fallback


def _positive_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) and number > 0.0 else fallback


def _unit_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return float(np.clip(number, 0.0, 1.0)) if np.isfinite(number) else fallback


def _rotation(value: Any, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        return fallback
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return fallback
    return result if all(np.isfinite(item) for item in result) else fallback  # type: ignore[return-value]


def _color4(value: Any) -> tuple[float, float, float, float]:
    try:
        raw = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raw = np.asarray((0.8, 0.8, 0.8, 1.0))
    if len(raw) < 3:
        raw = np.asarray((0.8, 0.8, 0.8, 1.0))
    if np.max(raw[: min(4, len(raw))]) > 1.0:
        raw = raw / 255.0
    alpha = float(raw[3]) if len(raw) > 3 else 1.0
    result = np.clip(np.asarray((raw[0], raw[1], raw[2], alpha)), 0.0, 1.0)
    return tuple(float(item) for item in result)  # type: ignore[return-value]


def _color3(value: Any) -> tuple[float, float, float]:
    color = _color4(value)
    return color[:3]


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _safe_filename(value: Any, fallback: str) -> str:
    try:
        return Path(value).name or fallback
    except (TypeError, ValueError, OSError):
        return fallback


def _brief_error(error: Exception) -> str:
    return str(error).strip() or error.__class__.__name__


__all__ = [
    "ComponentMetadata",
    "MaterialData",
    "MeshGroup",
    "MeshPrimitive",
    "MotherboardModel",
    "RaycastHit",
    "SUPPORTED_MODEL_FORMATS",
    "SceneModel",
    "SceneNode",
]
