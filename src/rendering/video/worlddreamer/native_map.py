# ruff: noqa: E722
# Extracted native raster helpers retain upstream exception behavior.
"""Unmodified selected WorldDreamer map primitives, imported without its training dataset.
Source: vendor/WorldDreamer/dataset/map_utils.py, DriveArena Apache-2.0.
The box adapter supplies already-centred corners, so box_center_shift is identity.
"""

import copy

import cv2
import geopandas as gpd
import numpy as np
import torch
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuscenes.utils.geometry_utils import view_points
from scipy.spatial.transform import Rotation as R
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union


def box_center_shift(box, origin):
    if tuple(origin) != (0.5, 0.5, 0.5):
        raise ValueError("Only already-centred boxes supported")
    return box


def visualize_bev_hdmap(
    gt_lines_instance,
    gt_labels_3d,
    canvas_size,
    num_classes=3,
    bound=[-50.0, 50.0],
    drivable_mask=None,
    nuplan=None,
):
    canvas = np.zeros((num_classes, *canvas_size, 3), dtype=np.uint8)
    for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d):
        pts = np.array(gt_line_instance)
        for p in pts:
            pp = ((p - bound[0]) / (bound[1] - bound[0]) * canvas_size[0]).astype(int)
            cv2.circle(canvas[int(gt_label_3d)], tuple(pp), 1, (1, 0, 0), -1)

        for i in range(len(pts) - 1):
            pp1 = ((pts[i] - bound[0]) / (bound[1] - bound[0]) * canvas_size[0]).astype(int)
            pp2 = ((pts[i + 1] - bound[0]) / (bound[1] - bound[0]) * canvas_size[0]).astype(int)
            cv2.line(canvas[int(gt_label_3d)], tuple(pp1), tuple(pp2), (1, 0, 0), 1)
    canvas = canvas[..., 0]

    if drivable_mask is not None:
        drivable_mask = drivable_mask[None, ...]
        drivable_mask = np.transpose(drivable_mask, (0, 2, 1))
        canvas = np.concatenate([canvas, drivable_mask], 0)
    canvas = np.transpose(canvas, (2, 1, 0))  # H, W, C
    # if nuplan:
    #     tmp = canvas[..., 1:] * 255
    #     cv2.imwrite("proj_map_nuplan.jpg", tmp)
    return canvas


def project_map_to_image_nuplan(
    gt_lines_instance,
    gt_labels_3d,
    intrinsic,
    extrinsic,
    num_classes=3,
    image=None,
    drivable_mask=None,
):
    z = 0
    canvas = np.zeros((num_classes, 900, 1600, 3), dtype=np.uint8)
    for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d):
        pts = torch.Tensor(gt_line_instance)

        dummy_pts = torch.cat([pts, torch.ones((pts.shape[0], 1)) * z], dim=-1)
        points_in_cam_cor = torch.matmul(
            extrinsic[:3, :3].T, (dummy_pts.T - extrinsic[:3, 3].reshape(3, -1))
        )
        points_in_cam_cor = points_in_cam_cor[:, points_in_cam_cor[2, :] > 0.1]
        if points_in_cam_cor.shape[1] > 0:
            points_on_image_cor = view_points(points_in_cam_cor, intrinsic, normalize=True)[:2, :].T
            points_on_image_cor = points_on_image_cor.astype(int)
        else:
            points_on_image_cor = []

        for i in range(len(points_on_image_cor) - 1):
            cv2.line(
                canvas[int(gt_label_3d)],
                tuple(points_on_image_cor[i]),
                tuple(points_on_image_cor[i + 1]),
                (1, 0, 0),
                4,
            )

        if image is not None:
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
            for i in range(len(points_on_image_cor) - 1):
                cv2.line(
                    image,
                    tuple(points_on_image_cor[i]),
                    tuple(points_on_image_cor[i + 1]),
                    colors[int(gt_label_3d)],
                    5,
                )

    if drivable_mask is not None:
        drivable_canvas = np.zeros((1, 900, 1600, 3), dtype=np.uint8)
        drivable_grids = np.where(drivable_mask > 0)
        drivable_grids = np.stack(drivable_grids, 1)
        drivable_pts = drivable_grids * 0.5 - 50.0 + 0.25
        # drivable_pts = drivable_pts[:,[1,0]]
        # drivable_pts[:,1] = -drivable_pts[:, 1]
        drivable_pts = torch.from_numpy(drivable_pts)

        dummy_pts = torch.cat(
            [drivable_pts, torch.ones((drivable_pts.shape[0], 1)) * z], dim=-1
        ).float()
        points_in_cam_cor = torch.matmul(
            extrinsic[:3, :3].T, (dummy_pts.T - extrinsic[:3, 3].reshape(3, -1))
        )
        points_in_cam_cor = points_in_cam_cor[:, points_in_cam_cor[2, :] > 0]
        if points_in_cam_cor.shape[1] > 1:
            intrinsic = intrinsic.float()
            points_on_image_cor = intrinsic[:3, :3] @ points_in_cam_cor
            points_on_image_cor = points_on_image_cor / (points_on_image_cor[-1, :].reshape(1, -1))
            points_on_image_cor = points_on_image_cor[:2, :].T
            points_on_image_cor = points_on_image_cor.int().numpy()

            for point in points_on_image_cor:
                cv2.circle(drivable_canvas[0], point, 20, (1, 0, 0), -1)

    if drivable_mask is not None:
        canvas = np.concatenate([canvas, drivable_canvas], 0)
    canvas = canvas[..., 0]
    canvas = np.transpose(canvas, (1, 2, 0))
    canvas = canvas[::4, ::4, :][1:, ...]  # 224, 400, 3
    # tmp = canvas[..., 1:] * 255
    # cv2.imwrite("proj_image_nuplan.jpg", tmp)
    return canvas


def project_box_to_image(
    gt_bboxes_3d, gt_labels_3d, transform, object_classes, image=None, cam_name=None
):

    bboxes_trans = box_center_shift(gt_bboxes_3d, (0.5, 0.5, 0.5))
    canvas = np.zeros((len(object_classes), 900, 1600, 3), dtype=np.uint8)
    if bboxes_trans is not None and len(bboxes_trans) > 0:
        corners = bboxes_trans.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate([corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1)
        transform = copy.deepcopy(transform.numpy()).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        gt_labels_3d = gt_labels_3d[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        gt_labels_3d = gt_labels_3d[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)

        for index in range(coords.shape[0]):
            for start, end in [
                (0, 3),
                (1, 2),
                (3, 2),
                (3, 7),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                try:
                    cv2.line(
                        canvas[int(gt_labels_3d[index])],
                        coords[index, start].astype(int),
                        coords[index, end].astype(int),
                        (1, 0, 0),
                        4,
                        cv2.LINE_AA,
                    )
                except:
                    pass
                if image is not None:
                    cv2.line(
                        image,
                        coords[index, start].astype(int),
                        coords[index, end].astype(int),
                        (120, 120, 120),
                        5,
                        cv2.LINE_AA,
                    )
            for start, end in [
                (0, 1),
                (0, 4),
                (1, 5),
                (4, 5),
            ]:
                try:
                    cv2.line(
                        canvas[int(gt_labels_3d[index])],
                        coords[index, start].astype(int),
                        coords[index, end].astype(int),
                        (2, 0, 0),
                        4,
                        cv2.LINE_AA,
                    )
                except:
                    pass
                if image is not None:
                    cv2.line(
                        image,
                        coords[index, start].astype(int),
                        coords[index, end].astype(int),
                        (23, 127, 245),
                        5,
                        cv2.LINE_AA,
                    )

    if image is not None:
        cv2.imwrite(f"./tmp/{cam_name}.png", image)

    canvas = canvas[..., 0]
    canvas = np.transpose(canvas, (1, 2, 0))
    canvas = canvas[::4, ::4, :][1:, ...]
    return canvas


def calculate_rotation(lidar2global_rotation):

    if isinstance(lidar2global_rotation, (np.ndarray, list)) and len(lidar2global_rotation) == 4:
        rotation = R.from_quat(lidar2global_rotation)
    # Check if the input is a rotation matrix (3x3 matrix)
    elif isinstance(lidar2global_rotation, (np.ndarray, list)) and np.shape(
        lidar2global_rotation
    ) == (3, 3):
        rotation = R.from_matrix(lidar2global_rotation.T)
    else:
        raise ValueError(
            "Invalid input: must be either a 3x3 rotation matrix or a quaternion with four elements."
        )

    ego_rotation_angle = rotation.as_euler("zxy", degrees=True)[0]

    return ego_rotation_angle


def nuplan_to_metadrive_vector(vector, nuplan_center=(0, 0)):
    "All vec in nuplan should be centered in (0,0) to avoid numerical explosion"
    vector = np.array(vector)
    vector -= np.asarray(nuplan_center)
    return vector


def extract_centerline(map_obj, nuplan_center):
    path = map_obj.baseline_path.discrete_path
    points = np.array(
        [nuplan_to_metadrive_vector([pose.x, pose.y], nuplan_center) for pose in path]
    )
    return points


def get_points_from_boundary(boundary, center):
    path = boundary.discrete_path
    points = [(pose.x, pose.y) for pose in path]
    points = nuplan_to_metadrive_vector(points, center)
    return points


def extract_map_features(map_api, center, radius=500):
    ret = {}
    np.seterr(all="ignore")
    # Center is Important !
    layer_names = [
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
    ]
    center_for_query = Point2D(*center)
    nearest_vector_map = map_api.get_proximal_map_objects(center_for_query, radius, layer_names)
    boundaries = map_api._get_vector_map_layer(SemanticMapLayer.BOUNDARIES)

    block_polygons = []
    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for block in nearest_vector_map[layer]:
            edges = (
                sorted(block.interior_edges, key=lambda lane: lane.index)
                if layer == SemanticMapLayer.ROADBLOCK
                else block.interior_edges
            )

            for index, lane_meta_data in enumerate(edges):
                if not hasattr(lane_meta_data, "baseline_path"):
                    continue
                if isinstance(lane_meta_data.polygon.boundary, MultiLineString):
                    boundary = gpd.GeoSeries(lane_meta_data.polygon.boundary).explode(
                        index_parts=True
                    )
                    sizes = []
                    for idx, polygon in enumerate(boundary[0]):
                        sizes.append(len(polygon.xy[1]))
                    points = boundary[0][np.argmax(sizes)].xy
                elif isinstance(lane_meta_data.polygon.boundary, LineString):
                    points = lane_meta_data.polygon.boundary.xy
                polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
                polygon = nuplan_to_metadrive_vector(polygon, nuplan_center=[center[0], center[1]])

                # According to the map attributes, lanes are numbered left to right with smaller indices being on the
                # left and larger indices being on the right.
                # @ See NuPlanLane.adjacent_edges()
                ret[lane_meta_data.id] = {
                    SD.TYPE: (
                        MetaDriveType.LANE_SURFACE_STREET
                        if layer == SemanticMapLayer.ROADBLOCK
                        else MetaDriveType.LANE_SURFACE_UNSTRUCTURE
                    ),
                    SD.POLYLINE: extract_centerline(lane_meta_data, center),
                    SD.ENTRY: [edge.id for edge in lane_meta_data.incoming_edges],
                    SD.EXIT: [edge.id for edge in lane_meta_data.outgoing_edges],
                    SD.LEFT_NEIGHBORS: (
                        [edge.id for edge in block.interior_edges[:index]]
                        if layer == SemanticMapLayer.ROADBLOCK
                        else []
                    ),
                    SD.RIGHT_NEIGHBORS: (
                        [edge.id for edge in block.interior_edges[index + 1 :]]
                        if layer == SemanticMapLayer.ROADBLOCK
                        else []
                    ),
                    SD.POLYGON: polygon,
                }
                if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR:
                    continue
                left = lane_meta_data.left_boundary
                if left.id not in ret:
                    # only broken line in nuPlan data
                    # line_type = get_line_type(int(boundaries.loc[[str(left.id)]]["boundary_type_fid"]))
                    line_type = MetaDriveType.LINE_BROKEN_SINGLE_WHITE
                    if line_type != MetaDriveType.LINE_UNKNOWN:
                        ret[left.id] = {
                            SD.TYPE: line_type,
                            SD.POLYLINE: get_points_from_boundary(left, center),
                        }

            if layer == SemanticMapLayer.ROADBLOCK:
                block_polygons.append(block.polygon)

    # corsswalk
    for area in nearest_vector_map[SemanticMapLayer.CROSSWALK]:
        if isinstance(area.polygon.exterior, MultiLineString):
            boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
            sizes = []
            for idx, polygon in enumerate(boundary[0]):
                sizes.append(len(polygon.xy[1]))
            points = boundary[0][np.argmax(sizes)].xy
        elif isinstance(area.polygon.exterior, LineString):
            points = area.polygon.exterior.xy
        polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
        polygon = nuplan_to_metadrive_vector(polygon, nuplan_center=[center[0], center[1]])
        ret[area.id] = {
            SD.TYPE: MetaDriveType.CROSSWALK,
            SD.POLYGON: polygon,
        }

    interpolygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]]
    boundaries = gpd.GeoSeries(unary_union(interpolygons + block_polygons)).boundary.explode(
        index_parts=True
    )

    if len(boundaries) > 0:
        for idx, boundary in enumerate(boundaries[0]):
            block_points = np.array(
                list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1]))
            )
            block_points = nuplan_to_metadrive_vector(block_points, center)
            id = "boundary_{}".format(idx)
            ret[id] = {
                SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_WHITE,
                SD.POLYLINE: block_points,
            }
    else:
        print("No boundaries found")
    np.seterr(all="warn")

    res = {}
    for idx, data in ret.items():
        if (
            data["type"] == MetaDriveType.LINE_SOLID_SINGLE_WHITE
            or data["type"] == MetaDriveType.LINE_BROKEN_SINGLE_WHITE
            or data["type"] == MetaDriveType.CROSSWALK
        ):
            res[idx] = data
        else:
            continue

    return res


def add_cross_zero_point(coords):
    # Check if the line crosses x=0 and add a point at x=0.1
    x_coords = coords[:, 0]
    if np.any(x_coords < 0) and np.any(x_coords > 0):
        # Find the index where the sign changes
        sign_changes = np.where(np.diff(np.sign(x_coords)))[0]
        for idx in sign_changes:
            # Linear interpolation to find the y-coordinate for specific x points
            x1, x2 = x_coords[idx], x_coords[idx + 1]
            y1, y2 = coords[idx, 1], coords[idx + 1, 1]

            for new_x in [1.5, 2, 2.5]:  # Define new x coordinates
                y_at_new_x = y1 + (y2 - y1) * (new_x - x1) / (x2 - x1)
                new_point = np.array([[new_x, y_at_new_x]])
                coords = np.insert(coords, idx + 1, new_point, axis=0)
                idx += 1  # Adjust index for inserted point

    return coords


def rotate_features(features, rotation_angle_degrees, type_to_int_map, boundary_box):
    """
    Rotate 'polyline' and 'polygon' coordinates for each feature in the dictionary,
    and clip to a given boundary without altering the original data.

    :param features: Dictionary of features with 'polyline' and 'polygon' coordinates.
    :param rotation_angle_degrees: The angle in degrees to rotate the coordinates.
    :param type_to_int_map: Map between integer type and string type.
    :param boundary_box: List of coordinates defining the clipping boundary.
    :return: Dictionary in the format of 'anns_results'.
    """
    temp_features = copy.deepcopy(features)

    # Prepare for transformations
    rotation_angle_radians = np.radians(rotation_angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle_radians), -np.sin(rotation_angle_radians)],
            [np.sin(rotation_angle_radians), np.cos(rotation_angle_radians)],
        ]
    )
    clipping_box = Polygon(boundary_box)

    # Initialize results dictionary
    results = {"gt_vecs_pts_loc": [], "gt_vecs_pts_num": [], "gt_vecs_label": []}

    for feature_id, data in temp_features.items():
        geometry = None

        if "polyline" in data:
            # Rotate and create LineString
            rotated_polyline = np.dot(data["polyline"], rotation_matrix)
            geometry = LineString(rotated_polyline)
        elif "polygon" in data:
            # Rotate and create Polygon
            rotated_polygon = np.dot(data["polygon"], rotation_matrix)
            geometry = LineString(Polygon(rotated_polygon).exterior.coords)

        # Perform clipping
        if geometry is not None:
            clipped_geometry = geometry.intersection(clipping_box)

            if not clipped_geometry.is_empty:
                # Handle Polygon conversion to LineString
                if isinstance(clipped_geometry, Polygon):
                    coords = np.array(clipped_geometry.exterior.coords)
                    coords = add_cross_zero_point(coords)
                    clipped_geometry = LineString(coords)

                # Check and handle MultiLineString or similar
                if isinstance(clipped_geometry, LineString):
                    coords = np.array(clipped_geometry.coords)
                    coords = add_cross_zero_point(coords)
                    clipped_geometry = LineString(coords)
                    results["gt_vecs_pts_loc"].append(clipped_geometry)
                    results["gt_vecs_pts_num"].append(len(clipped_geometry.coords))
                    results["gt_vecs_label"].append(type_to_int_map[data["type"]])
                elif hasattr(clipped_geometry, "geoms"):
                    for geom in clipped_geometry.geoms:
                        if isinstance(geom, (LineString, Polygon)):
                            if isinstance(geom, Polygon):
                                coords = np.array(geom.exterior.coords)
                                coords = add_cross_zero_point(coords)
                                geom = LineString(coords)
                                # geom = LineString(geom.exterior.coords)
                            coords = np.array(geom.coords)
                            coords = add_cross_zero_point(coords)
                            geom = LineString(coords)
                            results["gt_vecs_pts_loc"].append(geom)
                            results["gt_vecs_pts_num"].append(len(geom.coords))
                            results["gt_vecs_label"].append(type_to_int_map[data["type"]])
    return results
