"""CPU pinhole renderer of the physical planar world, never source-log context.

This deliberately primitive modality is for visual-interface experiments. Map
polylines are narrow symbolic stripes, not an invented drivable surface. Source
elevation is ignored; every object rests on z=0 and rig heights are above ground.
"""

from copy import deepcopy

import numpy as np

from driversi.contracts.scene import CYCLIST, LANE, PEDESTRIAN, VEHICLE, Scene
from driversi.contracts.visual import CameraRig, SensorBatch, WorldFrame

VEHICLE_COLOR = (70, 150, 225)
_COLORS = {VEHICLE: VEHICLE_COLOR, PEDESTRIAN: (222, 126, 137), CYCLIST: (210, 180, 81)}
_SKY = (163, 199, 222)
_GROUND = (75, 80, 73)
_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
          (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
_BRIGHTNESS = (.72, 1., .82, .88, .94, .78)


def _clip_depth(polygon: np.ndarray, boundary: float, keep_above: bool) -> np.ndarray:
    """Sutherland-Hodgman clipping in camera space, including crossing edges."""
    result = []
    previous = polygon[-1]
    previous_in = previous[2] >= boundary if keep_above else previous[2] <= boundary
    for current in polygon:
        current_in = current[2] >= boundary if keep_above else current[2] <= boundary
        if current_in != previous_in:
            fraction = (boundary - previous[2]) / (current[2] - previous[2])
            intersection = previous + fraction * (current - previous)
            intersection[2] = boundary
            result.append(intersection)
        if current_in:
            result.append(current)
        previous, previous_in = current, current_in
    return np.asarray(result, dtype=float).reshape(-1, 3)


def _triangle(vertices, intrinsic, color, instance, rgb, depth, instances):
    """Rasterize at pixel centers; interpolate inverse optical depth, not z."""
    projected = vertices @ intrinsic.T
    xy = projected[:, :2] / projected[:, 2:3]
    height, width = depth.shape
    xmin = max(0, int(np.ceil(np.min(xy[:, 0]) - .5)))
    xmax = min(width - 1, int(np.floor(np.max(xy[:, 0]) - .5)))
    ymin = max(0, int(np.ceil(np.min(xy[:, 1]) - .5)))
    ymax = min(height - 1, int(np.floor(np.max(xy[:, 1]) - .5)))
    if xmin > xmax or ymin > ymax:
        return
    a, b, c = xy
    area = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(area) < 1e-12:
        return
    yy, xx = np.mgrid[ymin:ymax+1, xmin:xmax+1].astype(float)
    xx += .5
    yy += .5
    w0 = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / area
    w1 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / area
    w2 = 1 - w0 - w1
    covered = (w0 >= -1e-10) & (w1 >= -1e-10) & (w2 >= -1e-10)
    inverse_z = w0 / vertices[0, 2] + w1 / vertices[1, 2] + w2 / vertices[2, 2]
    candidate = np.full(inverse_z.shape, np.inf)
    np.divide(1., inverse_z, out=candidate, where=covered & (inverse_z > 0))
    window = np.s_[ymin:ymax+1, xmin:xmax+1]
    current = depth[window]
    # Stable tie rule, including stripes on the analytical ground plane. A
    # farther fragment must never replace the nearer one outside roundoff.
    difference = np.full(candidate.shape, np.inf)
    np.subtract(candidate, current, out=difference,
                where=np.isfinite(candidate) & np.isfinite(current))
    equal = np.abs(difference) <= 1e-9
    visible = covered & ((candidate < current - 1e-9)
                         | (equal & (instance <= instances[window])))
    depth[window][visible] = candidate[visible]
    rgb[window][visible] = color
    instances[window][visible] = instance


def _polygon(vertices, camera, color, instance, rgb, depth, instances):
    polygon = _clip_depth(vertices, camera.near, True)
    if len(polygon) < 3:
        return
    polygon = _clip_depth(polygon, camera.far, False)
    for index in range(1, len(polygon) - 1):
        _triangle(polygon[[0, index, index+1]], camera.intrinsic,
                  color, instance, rgb, depth, instances)


class PrimitiveRenderer:
    """Read-only rendering of Scene actors at an explicitly supplied WorldFrame.

    A copied scene fixes physical IDs, dimensions and map geometry. WorldFrame
    supplies the only actor poses used; scene reference trajectories are ignored.
    Cameras are observer-relative and exclude their own body. Instance IDs use
    the scene order + 1 and must never enter a visual actor's policy input.
    """

    version = "primitive-cpu-v1"

    def __init__(self, scene: Scene, rig: CameraRig):
        scene.validate()
        self._scene = deepcopy(scene)
        self.rig = deepcopy(rig)
        for calibration in self.rig.cameras:
            calibration.intrinsic.setflags(write=False)
            calibration.camera_to_body.setflags(write=False)
        if any(camera.camera_to_body[2, 3] <= 0 for camera in rig.cameras):
            raise ValueError("Primitive cameras must be above the planar ground")
        self._indices = {actor: index for index, actor in enumerate(scene.actor_ids)}
        self._roads = []
        for index, kind in enumerate(scene.road_types):
            points = scene.road_points[scene.road_offsets[index]:scene.road_offsets[index+1], :2]
            half_width = .06 if kind == LANE else .09
            color = (150, 151, 144) if kind == LANE else (211, 205, 152)
            for start, end in zip(points[:-1], points[1:]):
                vector = end - start
                length = np.linalg.norm(vector)
                if length < 1e-9:
                    continue
                normal = np.asarray([-vector[1], vector[0]]) * half_width / length
                quad = np.zeros((4, 3))
                quad[:, :2] = [start+normal, end+normal, end-normal, start-normal]
                self._roads.append((quad, color))
        self._road_vertices = np.asarray([road[0] for road in self._roads]).reshape(-1, 4, 3)
        self._rays = []
        for camera in rig.cameras:
            yy, xx = np.mgrid[:camera.height, :camera.width]
            pixels = np.stack((xx+.5, yy+.5, np.ones_like(xx)), axis=-1)
            self._rays.append(pixels @ np.linalg.inv(camera.intrinsic).T)

    def _bodies(self, poses):
        bodies = []
        for index, (pose, size) in enumerate(zip(poses, self._scene.dimensions)):
            length, width, height = size
            corners = np.asarray([[-length/2, -width/2, 0], [length/2, -width/2, 0],
                                  [length/2, width/2, 0], [-length/2, width/2, 0],
                                  [-length/2, -width/2, height], [length/2, -width/2, height],
                                  [length/2, width/2, height], [-length/2, width/2, height]])
            cosine, sine = np.cos(pose[3]), np.sin(pose[3])
            rotation = np.asarray([[cosine, -sine], [sine, cosine]])
            corners[:, :2] = corners[:, :2] @ rotation.T + pose[:2]
            bodies.append((index, corners, np.asarray(_COLORS.get(int(self._scene.kinds[index]), (210,180,81)), float)))
        return bodies

    def _render_camera(self, pose, camera, rays, observer, bodies):
        cosine, sine = np.cos(pose[3]), np.sin(pose[3])
        body_rotation = np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
        rotation = body_rotation @ camera.camera_to_body[:3, :3]
        origin = body_rotation @ camera.camera_to_body[:3, 3] + [pose[0], pose[1], 0]
        rgb = np.empty((camera.height, camera.width, 3), dtype=np.uint8)
        rgb[:] = _SKY
        depth = np.full((camera.height, camera.width), np.inf)
        instances = np.zeros(depth.shape, dtype=np.int32)
        world_rays = rays @ rotation.T
        ground_z = np.full(depth.shape, np.inf)
        np.divide(-origin[2], world_rays[..., 2], out=ground_z,
                  where=world_rays[..., 2] < -1e-12)
        ground = (ground_z >= camera.near) & (ground_z <= camera.far)
        depth[ground] = ground_z[ground]
        rgb[ground] = _GROUND
        # Whole convex quads outside any frustum plane cannot cover a pixel.
        # Transform/cull all map segments together; keep the original draw order
        # and rasterizer for every surviving quad, including plane crossings.
        map_vertices = (self._road_vertices - origin) @ rotation
        projected = map_vertices @ camera.intrinsic.T
        z = map_vertices[..., 2]
        x, y = projected[..., 0], projected[..., 1]
        outside = ((z < camera.near - 1e-9).all(axis=1)
                   | (z > camera.far + 1e-9).all(axis=1)
                   | (x < -1e-9).all(axis=1)
                   | (x > camera.width*z + 1e-9).all(axis=1)
                   | (y < -1e-9).all(axis=1)
                   | (y > camera.height*z + 1e-9).all(axis=1))
        for index in np.flatnonzero(~outside):
            _polygon(map_vertices[index], camera, self._roads[index][1], 0, rgb, depth, instances)
        for index, vertices, base_color in bodies:
            if index == observer:
                continue
            camera_vertices = (vertices - origin) @ rotation
            if (np.max(camera_vertices[:, 2]) < camera.near
                    or np.min(camera_vertices[:, 2]) > camera.far):
                continue
            for face, brightness in zip(_FACES, _BRIGHTNESS):
                color = np.rint(base_color * brightness).astype(np.uint8)
                _polygon(camera_vertices[list(face)], camera, color, index+1, rgb, depth, instances)
        return rgb, depth, instances

    def render_batch(self, frame: WorldFrame, actor_ids: tuple[str, ...]) -> SensorBatch:
        """Batch the requested observers from exactly one frozen physical frame."""
        actor_ids = tuple(actor_ids)
        if (tuple(frame.actor_ids) != self._scene.actor_ids or not actor_ids
                or len(set(actor_ids)) != len(actor_ids)
                or any(actor not in self._indices for actor in actor_ids)):
            raise ValueError("World IDs must match the scene; observers must be unique physical actors")
        camera = self.rig.cameras[0]
        shape = (len(actor_ids), len(self.rig.cameras), camera.height, camera.width)
        rgb = np.empty((*shape, 3), dtype=np.uint8)
        depth = np.empty(shape, dtype=float)
        instances = np.empty(shape, dtype=np.int32)
        presence = self._scene.presence_at(frame.tick)
        if any(not presence[self._indices[actor]] for actor in actor_ids):
            raise ValueError("Absent participant cannot own a camera")
        bodies = [body for body in self._bodies(frame.poses) if presence[body[0]]]
        for actor_index, actor in enumerate(actor_ids):
            observer = self._indices[actor]
            for camera_index, (camera, rays) in enumerate(zip(self.rig.cameras, self._rays)):
                result = self._render_camera(frame.poses[observer], camera, rays, observer, bodies)
                rgb[actor_index, camera_index] = result[0]
                depth[actor_index, camera_index] = result[1]
                instances[actor_index, camera_index] = result[2]
        return SensorBatch(actor_ids, tuple(camera.camera_id for camera in self.rig.cameras),
                           rgb, depth, instances, np.ones(shape[:2], bool), frame.tick,
                           frame.episode_id, frame.branch_id, frame.generation)
