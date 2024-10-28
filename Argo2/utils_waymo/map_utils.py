'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

from shapely import LineString, Point, Polygon
import numpy as np
from utils_common.spline import Spline
import utils_common.helper_utils as utils
from utils_common.helper_utils import waymo_lane_type_converter, waymo_boundary_type_converter
import warnings
import glob
import os
import hashlib

from copy import deepcopy


SEGMENT_DEG = 3

segment_s = Spline(SEGMENT_DEG)


def preprocess_map_info(scenario):
    map_infos = {
        'lane': {},
        'road_line': {},
        'crosswalk': {},
    }

    for cur_data in scenario.map_features:
        cur_info = {'id': cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info['type'] = cur_data.lane.type  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane, 4: cross_walk

            cur_polyline = np.stack(
                [np.array([point.x, point.y]) for point in cur_data.lane.polyline], axis=0)
            cur_info['points'] = cur_polyline
            
            cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
            cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)
            cur_info['left_neighbors'] = [left_neighbor.feature_id for left_neighbor in cur_data.lane.left_neighbors]
            cur_info['right_neighbors'] = [right_neighbor.feature_id for right_neighbor in cur_data.lane.right_neighbors]
            
            cur_info['left_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': x.boundary_type  # roadline type
                } for x in cur_data.lane.left_boundaries
            ]
            cur_info['right_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': x.boundary_type  # roadline type
                } for x in cur_data.lane.right_boundaries
            ]
            
            if cur_polyline.shape[0] > 1: # lane should have at least two points     
                map_infos['lane'][cur_data.id] = cur_info

        elif cur_data.road_line.ByteSize() > 0:
            cur_info['type'] = cur_data.road_line.type

            cur_polyline = np.stack(
                [np.array([point.x, point.y]) for point in cur_data.road_line.polyline], axis=0)
            cur_info['points']=cur_polyline

            map_infos['road_line'][cur_data.id] = cur_info

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info['type'] = cur_data.road_edge.type

            cur_polyline = np.stack([np.array([point.x, point.y]) for point in cur_data.road_edge.polyline], axis=0)
            cur_info['points'] = cur_polyline

            map_infos['road_line'][cur_data.id] = cur_info

        elif cur_data.crosswalk.ByteSize() > 0:
            cur_info['type'] = 4 # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane, 4: cross_walk
            
            cur_polyline = np.stack([np.array([point.x, point.y]) for point in cur_data.crosswalk.polygon], axis=0)
            edge_1, edge_2 = find_edges(cur_polyline)
            cur_info['edge_1'] = edge_1
            cur_info['edge_2'] = edge_2
            
            map_infos['crosswalk'][cur_data.id] = cur_info

        else:
            continue
            
    return map_infos


def get_map_features(scenario, origin=None, R=None, city_map = None):
    map_infos = preprocess_map_info(scenario)
    
    # initialization
    lane_boundary_cps = [] #np.zeros((num_lanes, 3, 4, 2), dtype=float)
    lane_boundary_type = [] #np.zeros((num_lanes, 2), dtype=np.uint8)
    lane_type = [] #np.zeros((num_lanes), dtype=np.uint8)
    lane_is_intersection = [] #np.zeros(num_lanes, dtype=np.uint8)

    for lane_id, lane in map_infos['lane'].items():
        pts = lane['points']
        hash_id = hashlib.md5(pts.tostring()).hexdigest()

        cps_list = []

        if city_map is not None and hash_id in city_map.keys():
            cps_list = city_map[hash_id]
        else:
            recurrent_fit_line(pts=pts, cps_list=cps_list, degree=SEGMENT_DEG)
            city_map[hash_id] = deepcopy(cps_list)

        lane_boundary_cps = lane_boundary_cps+ cps_list
        l_type = waymo_lane_type_converter(lane['type'])
        lane_type_list = [l_type for _ in range(len(cps_list))]
        lane_type = lane_type + lane_type_list

    num_lanes = len(lane_boundary_cps)

     # initialization
    cw_boundary_cps = [] #np.zeros((num_cross_walks, 3, self.PATH_DEG+1, 2), dtype=np.float32)
    cw_boundary_type = [] #np.zeros((num_cross_walks, 2), dtype=np.uint8)
    cw_type = [] #np.zeros((num_cross_walks), dtype=np.uint8)                           

    for cw_id, cw in map_infos['crosswalk'].items():
        edge_1 = cw['edge_1']
        edge_2 = cw['edge_2']
        center = (edge_1 + edge_2)/2.
        
        cw_cps = fit_line(center, degree = SEGMENT_DEG, use_borgespastva = False, num_sample_point=4)
        cw_cps_reverse = cw_cps[::-1]
        cw_boundary_cps.append(cw_cps)
        cw_boundary_cps.append(cw_cps_reverse)

        cw_type.append(waymo_lane_type_converter(cw['type']))
        cw_type.append(waymo_lane_type_converter(cw['type']))

    num_cws = len(cw_boundary_cps)

    lane_segment_ids = np.zeros(num_lanes, dtype=np.uint32)
    lane_boundary_cps = np.array(lane_boundary_cps, dtype=np.float32)

    lane_boundary_cps = (lane_boundary_cps - origin) @ R 
    lane_boundary_cps = np.concatenate([np.zeros((num_lanes, 2, SEGMENT_DEG+1, 2)), lane_boundary_cps[:, None, :, :]], axis = 1)
    lane_boundary_type = np.zeros((num_lanes, 2), dtype=np.uint8)
    lane_type = np.array(lane_type, dtype=np.uint8)
    lane_is_intersection = np.zeros(num_lanes, dtype=np.uint8)

    cross_walk_ids = np.zeros(num_cws, dtype=np.uint32)
    cw_boundary_cps = np.array(cw_boundary_cps, dtype=np.float32)
    if num_cws >0:
        cw_boundary_cps = (cw_boundary_cps-origin)@R
        cw_boundary_cps = np.concatenate([np.zeros((num_cws, 2, SEGMENT_DEG+1, 2)), cw_boundary_cps[:, None, :, :]], axis = 1)
    cw_boundary_type = np.zeros((num_cws, 2),  dtype=np.uint8)
    cw_type = np.array(cw_type, dtype=np.uint8)


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
        'num_cws': num_cws,
        'PATH_DEG': SEGMENT_DEG,
        'origin': origin, # [2]
        'ro_mat': R # [2, 2]
    }                
    
    return map_data
                                                       
            
################################ centerline ###########################################
def recurrent_fit_line(pts, cps_list, degree, current_iter =0, max_iter = 3):
    num_pts = pts.shape[0]
    lane_cps = fit_line(pts, degree=degree, use_borgespastva=True)

    if current_iter == max_iter:
        cps_list.append(lane_cps)
        return

    fit_error = np.linalg.norm(pts[[0, -1]] - lane_cps[[0, -1]], axis=-1)

    if np.max(fit_error) > 0.1 and num_pts >= 8:
        recurrent_fit_line(pts[:((num_pts // 2) +1)], cps_list, degree=degree, current_iter=current_iter + 1)
        recurrent_fit_line(pts[(num_pts // 2):], cps_list, degree=degree, current_iter=current_iter + 1)
    else:
        cps_list.append(lane_cps)
        return
    

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
        l = resample_line(clean_lines(line)[0], num_sample_point) #av2.geometry.interpolate.interp_arc(num_sample_point, line)
        
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

def find_edges(xy):
    if xy.shape[0] != 4:
        polygon = Polygon(xy)
        rect = polygon.minimum_rotated_rectangle
        x, y = rect.boundary.xy
        xy = np.concatenate([np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)], axis=1)
    
    dist_1 = np.linalg.norm(xy[0] - xy[1], axis=-1)
    dist_2 = np.linalg.norm(xy[1] - xy[2], axis=-1)
    
    if dist_1 >= dist_2:
        return xy[:2], xy[[-1, -2]]
    else:
        return xy[[1,2]], xy[[0,3]]