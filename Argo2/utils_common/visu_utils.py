'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
from scipy.stats.distributions import chi2
import numpy as np
from av2.map.lane_segment import LaneType, LaneMarkType
import av2


def draw_confidence_ellipse(mean, cov, ax, **kwarg):
    '''
    Draw the covariance ellipse with 2-Sigma confidence.
    mean: mean of multivariate Normal Distribution
    cov: covariance of multivariate Normal Distribution
    '''
    
    d,u = np.linalg.eigh(cov)
    
    confidence = chi2.ppf(0.95,2)
    height, width = 2*np.sqrt(confidence*d)
    angle = np.degrees(np.arctan2(u[1,-1], u[0,-1]))
    
    #ax.scatter(*mean, s=100, marker='x', **kwarg)
    
    ellipse = Ellipse(
        xy=mean, 
        width=width, 
        height=height, 
        angle = angle,
        fill = False,
        **kwarg
    )
    
    ax.add_artist(ellipse)
    

def lane_mark_type_to_config(lane_mark_type):
    color, typ, lw = None, None, 1
    if "WHITE" in lane_mark_type:
        color = "grey"
    elif "YELLOW" in lane_mark_type:
        color = "orange"
    elif "BLUE" in lane_mark_type:
        color = "blue"
    else:
        color = "black"
        
    if "SOLID_DASH" in lane_mark_type:
        typ = "-."
    elif "DASH_SOLID" in lane_mark_type:
        typ = "-."
    elif "DOUBLE_SOLID" in lane_mark_type:
        typ, lw = "x", 3
    elif "DOUBLE_DASH" in lane_mark_type:
        typ, lw = "x", 3
    elif "SOLID" in lane_mark_type:
        typ = '-'
    elif "DASHED" in lane_mark_type:
        typ = '--'
    else:
        typ, lw = '^', 3
        
        
    return color, typ, lw

def lane_type_to_config(lane_type):
    color = None
    if lane_type == "VEHICLE":
        color = "lightgrey"
    elif lane_type == "BIKE":
        color = "red"
    elif lane_type == "BUS":
        color = "blue"
    elif lane_type == "PEDESTRIAN":
        color = "green"
    
    return color
    
def visualise_map(avm, ax, origin = None, R = None):
    all_lane_ids = avm.get_scenario_lane_segment_ids()
    
    if origin is not None and R is not None:
        polygons = [(avm.get_lane_segment_polygon(l_id)[:, :2] - origin) @ R for l_id in all_lane_ids]
    else:
        polygons = [avm.get_lane_segment_polygon(l_id)[:, :2] for l_id in all_lane_ids]
    #centerlines = np.array([avm.get_lane_segment_centerline(l_id)[:, :2] for l_id in all_lane_ids])

    
    for i, lane_id in enumerate(all_lane_ids):
        l_bound = av2.geometry.interpolate.interp_arc(8, avm.vector_lane_segments[lane_id].left_lane_boundary.xyz[:, :2])
        r_bound = av2.geometry.interpolate.interp_arc(8, avm.vector_lane_segments[lane_id].right_lane_boundary.xyz[:, :2])
        
        if lane_id == 239092259:
            l_bound = avm.vector_lane_segments[lane_id].left_lane_boundary.xyz[:, :2]
            r_bound = avm.vector_lane_segments[lane_id].right_lane_boundary.xyz[:, :2]
            # print(l_bound.shape, r_bound.shape)
        if origin is not None and R is not None:
            l_bound = (l_bound - origin) @ R
            r_bound = (r_bound - origin) @ R
            
        
        l_color, l_type, llw = lane_mark_type_to_config(avm.vector_lane_segments[lane_id].left_mark_type.value)
        r_color, r_type, rlw = lane_mark_type_to_config(avm.vector_lane_segments[lane_id].right_mark_type.value)
        if lane_id == 239092259:
            ax.plot(*l_bound.T, l_type, color = 'r', markersize = llw)
            ax.plot(*r_bound.T, r_type, color = 'r', markersize = rlw)
        else:
            ax.plot(*l_bound.T, l_type, color = l_color, markersize = llw)
            ax.plot(*r_bound.T, r_type, color = r_color, markersize = rlw)
        
        road_color = lane_type_to_config(avm.vector_lane_segments[lane_id].lane_type.value)
        polygon = Polygon(polygons[i], True, edgecolor = None, facecolor=road_color, alpha = 0.2)
        ax.add_patch(polygon)
    
    
    for i, cw in enumerate(avm.get_scenario_ped_crossings()):
        edge1 = cw.edge1.xyz[:, :2]
        edge2 = cw.edge2.xyz[:, :2]
        cw_polygon = np.concatenate([edge1, edge2[::-1], edge1[[0], :]], axis = 0)

        if origin is not None and R is not None:
            cw_polygon = (cw_polygon -origin) @ R
            
        cw_polygon = Polygon(cw_polygon, True, edgecolor = None, facecolor="green", alpha = 0.2)
        ax.add_patch(cw_polygon)