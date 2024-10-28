'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

from av2.map.map_api import ArgoverseStaticMap
from shapely import LineString, Point, Polygon
import numpy as np
from pathlib import Path
import glob
import os
import hashlib

from utils_common.spline import Spline
import utils_common.helper_utils as utils


class MapInterpreter():
    def __init__(self, 
                 src_file: str, 
                 path_degree: int):
        map_file = glob.glob(os.path.join(src_file, '*.json*'))[0]
        self.avm = ArgoverseStaticMap.from_json(Path(map_file))
        self.PATH_DEG = path_degree
        self.s = Spline(self.PATH_DEG)
        self.curvature_tol = 2
        
        self.all_polygon_dict, self.vehicle_polygon_dict, self.bike_polygon_dict, self.bus_polygon_dict , self.pedes_polygon_dict = {}, {}, {}, {}, {}
        self.all_lane_ids, self.vehicle_lane_ids, self.bike_lane_ids, self.bus_lane_ids, self.pedes_lane_ids = [], [], [], [], []
        self.lane_center_dict = {}
        self.adjacent_lane_dict = {}
        
        self._generate_polygons_dict_()
        
    def _generate_polygons_dict_(self):
        self.all_lane_ids = self.avm.get_scenario_lane_segment_ids()
        self.vehicle_lane_ids = [l_id for l_id in self.all_lane_ids if utils.POLYGON_TYPES.index(self.avm.vector_lane_segments[l_id].lane_type.value) == 0]
        self.bike_lane_ids = [l_id for l_id in self.all_lane_ids if utils.POLYGON_TYPES.index(self.avm.vector_lane_segments[l_id].lane_type.value) == 1 ]
        self.bus_lane_ids = [l_id for l_id in self.all_lane_ids if utils.POLYGON_TYPES.index(self.avm.vector_lane_segments[l_id].lane_type.value) == 2]
        self.pedes_lane_ids = [transform_cw_id(cw_id, 1) for cw_id in self.avm.vector_pedestrian_crossings.keys()] + [transform_cw_id(cw_id, 2) for cw_id in self.avm.vector_pedestrian_crossings.keys()]
        
        self.all_polygon_dict = {l_id: Polygon(self.avm.get_lane_segment_polygon(l_id)) for l_id in self.all_lane_ids}
        self.vehicle_polygon_dict = {l_id: Polygon(self.avm.get_lane_segment_polygon(l_id)) for l_id in self.vehicle_lane_ids}
        self.bike_polygon_dict = {l_id: Polygon(self.avm.get_lane_segment_polygon(l_id)) for l_id in self.bike_lane_ids}
        self.bus_polygon_dict = {l_id: Polygon(self.avm.get_lane_segment_polygon(l_id)) for l_id in self.bus_lane_ids}
        pedes_polygon_dict_1 = {transform_cw_id(cw_id, 1): Polygon(self.avm.vector_pedestrian_crossings[cw_id].polygon) for cw_id in self.avm.vector_pedestrian_crossings.keys()}
        pedes_polygon_dict_2 = {transform_cw_id(cw_id, 2): Polygon(self.avm.vector_pedestrian_crossings[cw_id].polygon) for cw_id in self.avm.vector_pedestrian_crossings.keys()}
        pedes_polygon_dict_1.update(pedes_polygon_dict_2)
        self.pedes_polygon_dict = pedes_polygon_dict_1
        
        self.lane_center_dict = {l_id: LineString(self.avm.get_lane_segment_centerline(l_id)[:, :2]) for l_id in self.all_lane_ids}
        
        cw_center_dict_1, cw_center_dict_2 = {}, {}
        for cw_id in self.avm.vector_pedestrian_crossings.keys():
            edge = np.mean(np.array(self.avm.vector_pedestrian_crossings[cw_id].get_edges_2d()), axis=0)
            cw_center_dict_1[transform_cw_id(cw_id, 1)] = LineString(edge)
            cw_center_dict_2[transform_cw_id(cw_id, 2)] = LineString(edge[::-1])
 
        self.lane_center_dict.update(cw_center_dict_1)
        self.lane_center_dict.update(cw_center_dict_2)
        
    
################################ Map Features ################################################
    def get_map_features(self, origin = None, R = None, city_map = None):
        lane_segment_ids = self.all_lane_ids
        num_lanes = len(lane_segment_ids)

        # initialization
        lane_boundary_cps = np.zeros((num_lanes, 3, self.PATH_DEG+1, 2), dtype=float)
        lane_boundary_type = np.zeros((num_lanes, 2), dtype=np.uint8)
        lane_type = np.zeros((num_lanes), dtype=np.uint8)
        lane_is_intersection = np.zeros(num_lanes, dtype=np.uint8)
        
        outlier_lane_ids = []
        
        for lane_segment in self.avm.get_scenario_lane_segments():
            lane_segment_idx = lane_segment_ids.index(lane_segment.id)
            left_bound = self.avm.vector_lane_segments[lane_segment.id].left_lane_boundary
            right_bound = self.avm.vector_lane_segments[lane_segment.id].right_lane_boundary
            
            hash_id = hashlib.md5(np.concatenate([left_bound.xyz, right_bound.xyz], axis = 0).tostring()).hexdigest()
            
            if city_map is None:
                lane_boundary_cps[lane_segment_idx, 0] = fit_line(left_bound.xyz[:, :2], degree = self.PATH_DEG, use_borgespastva=True if left_bound.xyz.shape[0] > 6 else False)
                lane_boundary_cps[lane_segment_idx, 1] = fit_line(right_bound.xyz[:, :2], degree = self.PATH_DEG, use_borgespastva=True if right_bound.xyz.shape[0] > 6 else False)
                lane_boundary_cps[lane_segment_idx, 2] = fit_line(self.avm.get_lane_segment_centerline(lane_segment.id)[:, :2], degree = self.PATH_DEG, no_clean = True, use_borgespastva=True\
                                                                 if left_bound.xyz.shape[0] > 6 or right_bound.xyz.shape[0] > 6 else False)
            elif hash_id in city_map.keys():
                lane_boundary_cps[lane_segment_idx] = city_map[hash_id]
            else:
                lane_boundary_cps[lane_segment_idx, 0] = fit_line(left_bound.xyz[:, :2], degree = self.PATH_DEG, use_borgespastva=True if left_bound.xyz.shape[0] > 6 else False)
                lane_boundary_cps[lane_segment_idx, 1] = fit_line(right_bound.xyz[:, :2], degree = self.PATH_DEG, use_borgespastva=True if right_bound.xyz.shape[0] > 6 else False)
                lane_boundary_cps[lane_segment_idx, 2] = fit_line(self.avm.get_lane_segment_centerline(lane_segment.id)[:, :2], degree = self.PATH_DEG, no_clean = True, use_borgespastva=True\
                                                                 if (left_bound.xyz.shape[0] > 6 or right_bound.xyz.shape[0] > 6) else False)
                city_map[hash_id] = lane_boundary_cps[lane_segment_idx]
            
            lane_boundary_type[lane_segment_idx, 0] = utils.POINT_TYPES.index(self.avm.vector_lane_segments[lane_segment.id].left_mark_type.value)
            lane_boundary_type[lane_segment_idx, 1] = utils.POINT_TYPES.index(self.avm.vector_lane_segments[lane_segment.id].right_mark_type.value)
            
            lane_type[lane_segment_idx] = utils.POLYGON_TYPES.index(lane_segment.lane_type.value)
            lane_is_intersection[lane_segment_idx] = utils.POLYGON_IS_INTERSECTIONS.index(lane_segment.is_intersection)
            
            # Check curvature
            if np.max(self.s.curvature(lane_boundary_cps[lane_segment_idx, 0], return_abs = True)) > self.curvature_tol or \
               np.max(self.s.curvature(lane_boundary_cps[lane_segment_idx, 1], return_abs = True)) > self.curvature_tol:
                outlier_lane_ids.append(lane_segment.id)
        
        
        cross_walk_ids = list(self.avm.vector_pedestrian_crossings.keys())
        num_cross_walks = len(cross_walk_ids) * 2
        
        # initialization
        cw_boundary_cps = np.zeros((num_cross_walks, 3, self.PATH_DEG+1, 2), dtype=np.float32)
        cw_boundary_type = np.zeros((num_cross_walks, 2), dtype=np.uint8)
        cw_type = np.zeros((num_cross_walks), dtype=np.uint8)

        
        for crosswalk in self.avm.get_scenario_ped_crossings():
            crosswalk_idx = cross_walk_ids.index(crosswalk.id)
            edge1 = crosswalk.edge1.xyz[:, :2]
            edge2 = crosswalk.edge2.xyz[:, :2]
            
            edge1_cps = fit_line(edge1, degree = self.PATH_DEG, use_borgespastva=False, num_sample_point=4)
            edge2_cps = fit_line(edge2, degree = self.PATH_DEG, use_borgespastva=False, num_sample_point=4)
            
            edge1_cps_reverse = edge1_cps[::-1, :]
            edge2_cps_reverse = edge2_cps[::-1, :]
            
            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2
            
            if side_to_directed_lineseg((edge1[0] + edge1[-1]) / 2, start_position, end_position) == 'LEFT':
                cw_boundary_cps[crosswalk_idx, 0] = edge1_cps
                cw_boundary_cps[crosswalk_idx, 1] = edge2_cps
                cw_boundary_cps[crosswalk_idx, 2] = (edge1_cps + edge2_cps) / 2.

                cw_boundary_cps[crosswalk_idx + len(cross_walk_ids), 0] = edge2_cps_reverse
                cw_boundary_cps[crosswalk_idx + len(cross_walk_ids), 1] = edge1_cps_reverse
                cw_boundary_cps[crosswalk_idx + len(cross_walk_ids), 2] = (edge1_cps_reverse + edge2_cps_reverse)/2.
            else:
                cw_boundary_cps[crosswalk_idx, 0] = edge2_cps
                cw_boundary_cps[crosswalk_idx, 1] = edge1_cps
                cw_boundary_cps[crosswalk_idx, 2] = (edge1_cps + edge2_cps) / 2.

                cw_boundary_cps[crosswalk_idx + len(cross_walk_ids), 0] = edge1_cps_reverse
                cw_boundary_cps[crosswalk_idx + len(cross_walk_ids), 1] = edge2_cps_reverse
                cw_boundary_cps[crosswalk_idx + len(cross_walk_ids), 2] = (edge1_cps_reverse + edge2_cps_reverse)/2.
            
            cw_boundary_type[crosswalk_idx, 0] = utils.POINT_TYPES.index('CROSSWALK')
            cw_boundary_type[crosswalk_idx, 1] = utils.POINT_TYPES.index('CROSSWALK')
            cw_boundary_type[crosswalk_idx + len(cross_walk_ids), 0] = utils.POINT_TYPES.index('CROSSWALK')
            cw_boundary_type[crosswalk_idx + len(cross_walk_ids), 1] = utils.POINT_TYPES.index('CROSSWALK')
            
            cw_type[crosswalk_idx] = utils.POLYGON_TYPES.index('PEDESTRIAN')
            cw_type[crosswalk_idx + len(cross_walk_ids)] = utils.POLYGON_TYPES.index('PEDESTRIAN')
        
        if origin is not None and R is not None:
            lane_boundary_cps = (lane_boundary_cps - origin) @ R
            cw_boundary_cps = (cw_boundary_cps - origin) @ R
            
        map_data = {
            'map_lane_ids': lane_segment_ids,
            'map_lane_boundary_cps': lane_boundary_cps,
            'map_lane_boundary_type': lane_boundary_type,
            'map_lane_type': lane_type,
            'map_lane_is_intersection': lane_is_intersection,
            'map_cw_ids': cross_walk_ids,
            'map_cw_boundary_cps': cw_boundary_cps,
            'map_cw_boundary_type': cw_boundary_type,
            'map_cw_type': cw_type,
            'num_lanes': num_lanes,
            'num_cws': num_cross_walks,
            'PATH_DEG': self.PATH_DEG,
            'origin': origin, # [2]
            'ro_mat': R # [2, 2]
        }

        return map_data, outlier_lane_ids
    
################################ centerline ###########################################
def fit_line(line: np.ndarray,
             degree: int,
             use_borgespastva = False,
             num_sample_point = 12,
             maxiter = 2,
             no_clean = False):
    '''
    fit line and find control points.
    
    parameter:
        - line: [N, 2]
        
    return:
        - resampled (interpolated) line [deg + 1,2]
    '''
    if line.shape[0] == 2 or no_clean:
        l = resample_line(line, num_sample_point)
    else:
        l = resample_line(clean_lines(line)[0], num_sample_point)
        
    s = Spline(degree)
    cps = s.find_control_points(l)
    
    if use_borgespastva:
        t0 = s.initial_guess_from_control_points(l, cps)
        t, converged , errors = s.borgespastva(l, k = degree, t0 = t0, maxiter=maxiter)
        cps = s.find_control_points(l, t=t)
        
    return cps


def resample_line(line: np.ndarray, num_sample_point = 12):
    '''
    resample (interpolate) line with equal distance.
    
    parameter:
        - line: [N, 2]
        
    return:
        - resampled (interpolated) line [M,2]
    '''

    ls = LineString(line)
    s0 = 0
    s1 = ls.length
    
    return np.array([
        ls.interpolate(s).coords.xy
        for s in np.linspace(s0, s1, num_sample_point)
    ]).squeeze()

def resample_path(line: np.ndarray, x0 = None, max_dist=200):
    '''
    resample (interpolate) line with equal distance.
    
    parameter:
        - line: [N, 2]
        - x0: start position
        
    return:
        - resampled (interpolated) line [M,2]
    '''

    ls = LineString(line)
    if x0 is None:
        s0 = 0
    else:
        s0 = ls.project(Point(*x0))
    s1 = min(ls.length, s0 + max_dist)
    
    if s0 == s1:
        s0 = 0
        s1 = min(ls.length, s0 + max_dist)
        
    return np.array([
        ls.interpolate(s).coords.xy
        for s in np.linspace(s0, s1, max(int(s1 - s0), 11))
    ]).squeeze()

def clean_lines(lines):
    '''
    clean line points, which go backwards.
    
    parameter:
        - lines: list of lines with shape [N, 2]
        
    return:
        - cleaned list of lines with shape [M, 2]
    '''
    cleaned_lines = []
    if not isinstance(lines, list):
        lines = [lines]
    for candidate in lines:
        # remove duplicate points
        ds = np.linalg.norm(np.diff(candidate, axis=0), axis=-1) > 0.05
        keep = np.block([True, ds])
        
        cleaned = candidate[keep, :]
        
        # remove points going backward
        dx, dy = np.diff(cleaned, axis=0).T
        dphi = np.diff(np.unwrap(np.arctan2(dy, dx)))
        
        keep = np.block([True, dphi < (np.pi / 2), True])
        
        cleaned = cleaned[keep, :]
        
        cleaned_lines.append(cleaned)
        
    return cleaned_lines

def angle_between_vectors(v1, v2):
    return np.arccos(v1@v2.T / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

def side_to_directed_lineseg(
        query_point,
        start_point,
        end_point) -> str:
    cond = np.cross((end_point - start_point), (query_point - start_point))
    if cond > 0:
        return 'LEFT'
    elif cond < 0:
        return 'RIGHT'
    else:
        return 'CENTER'
    

def line_string_to_xy(ls):
    x, y = ls.coords.xy
    return np.vstack([x,y]).T

def transform_cw_id(cw_id, additional_id):
    return int(str(cw_id) + str(cw_id) + str(additional_id))
