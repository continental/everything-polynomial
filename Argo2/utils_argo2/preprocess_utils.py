'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import sys, os
import numpy as np
import json
import pickle
import multiprocessing
import glob
import pandas as pd
from tqdm.auto import tqdm
import warnings
from scipy.linalg import block_diag

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.map.map_api import ArgoverseStaticMap

from utils_common.helper_utils import OBJECT_TYPES, TRACK_CATEGORIES
import utils_argo2.tracking_utils as tracking_utils 
import utils_argo2.map_utils as map_utils 

sys.path.append('../cmotap/continental-multi-object-tracker-and-predictor/')
from src.cmotap.trajectory import Trajectory 
from src.cmotap.basisfunctions.bernsteinpolynomials import BernsteinPolynomials
from src.cmotap.motionmodels.trajectory_motion import TrajectoryMotion
from src.cmotap.observationmodels.trajectory_observation import TrajectoryObservation

from src.cmotap import utils

# Load prior parameters
with open('priors_argo2/vehicle/vehicle_5s.json', "r") as read_file:
    prior_vechicle = json.load(read_file)
    
with open('priors_argo2/cyclist/cyclist_5s.json', "r") as read_file:
    prior_cyclist = json.load(read_file)
    
with open('priors_argo2/pedestrian/pedestrian_5s.json', "r") as read_file:
    prior_pedestrian = json.load(read_file)
    
with open('priors_argo2/ego/ego_5s.json', "r") as read_file:
    prior_ego = json.load(read_file)


HIST_TIMESCALE = 4.9
HIST_DEGREE = 5 # According to AIC in https://arxiv.org/abs/2211.01696v4
PATH_DEGREE = 3 # Degree for lane segments and cross walks
SPACEDIM = 2
HIST_LEN = 50
MIN_OBS_LEN = 1

BASIS = BernsteinPolynomials(HIST_DEGREE)
TRAJ = Trajectory(basisfunctions=BASIS, spacedim=SPACEDIM, timescale=HIST_TIMESCALE)
                    
outlier_lane_dict = {}


def track_ego_trajectory(ego_traj_points, timestamps):
    assert ego_traj_points.shape[0] <= HIST_LEN
    assert ego_traj_points.shape[0] == len(timestamps)
    
    FUTUREPRIOR, HISTORYPRIOR = tracking_utils.get_prior(degree=HIST_DEGREE, timescale=HIST_TIMESCALE, prior_data=prior_ego)

    Q = np.kron(np.diag([0, 0, 0, .3, .2, .1])**2, np.eye(SPACEDIM))
    motion_prior = np.eye(BASIS.size * SPACEDIM) * 200000 # uninformative prior for motion   

    ego_state = None
    last_timestamp = None
    
    
    pos_obs_cov = tracking_utils.build_ego_observation_noise_covariance(degree = HIST_DEGREE, prior_data = prior_ego)
    
    for i, (timestamp, ego_traj_point) in enumerate(zip(timestamps, ego_traj_points)):
        ego_pos = ego_traj_point[:2]
        ego_vel = ego_traj_point[-2:]
        ego_heading = ego_traj_point[2]
        rot = np.array([[np.cos(ego_heading), -np.sin(ego_heading)],
                        [np.sin(ego_heading), np.cos(ego_heading)]])
        
        vel_obs_cov = np.diag([0.5, 0.1])**2
        vel_obs_cov = rot @ vel_obs_cov @ rot.T

        
        if ego_state is not None:
            # Prediction Step
            dt = timestamp - last_timestamp

            MM = TrajectoryMotion(TRAJ, Q, Prior=motion_prior)

            predicted_ego_state = ego_state.predict(MM, dt)

            # Update Step
            OM = TrajectoryObservation(TRAJ, t=HIST_TIMESCALE, derivatives=[[0,1]], R=[block_diag(pos_obs_cov, vel_obs_cov)])
            ego_state = predicted_ego_state.update(np.concatenate([ego_pos, ego_vel]), OM)

        else:
            ego_state = tracking_utils.get_initial_state_ego(ego_pos, ego_vel, HISTORYPRIOR, TRAJ, BASIS, HIST_TIMESCALE)
        
        last_timestamp = timestamp
        
    return np.array(ego_state.x, dtype = np.float32).reshape(-1,2),  np.array(ego_state.P, dtype = np.float32)


def track_obj_trajectory(obj_traj_points, ego_traj_points, timestamps, object_type):
    assert len(timestamps) <= HIST_LEN
    assert len(timestamps) == obj_traj_points.shape[0] == ego_traj_points.shape[0]
    
    prior_data = None
    if object_type == 0: # vehicle
        prior_data = prior_vechicle
    elif object_type == 1: # pedestrian
        prior_data = prior_pedestrian
    elif object_type == 3: # cyclist
        prior_data = prior_cyclist
    elif object_type == 2: # motor_cyclist
        prior_data = prior_cyclist
    elif object_type == 8: # riderless_bicycle
        prior_data = prior_cyclist
    else: # TODO: what is the prior for unknown objects?
        prior_data = prior_vechicle
    
    FUTUREPRIOR, HISTORYPRIOR = tracking_utils.get_prior(degree=HIST_DEGREE, timescale=HIST_TIMESCALE, prior_data=prior_data)
    
    Q = np.kron(np.diag([0, 0, 0, .3, .2, .1])**2, np.eye(SPACEDIM))
    
    motion_prior = np.eye(BASIS.size * SPACEDIM) * 200000 # uninformative prior for motion  
    obj_state = None
    last_timestamp = None    
    
    
    pos_obs_cov_temp = tracking_utils.build_obj_observation_noise_covariance_batch(ego_pos=ego_traj_points[:,:2], 
                                                                                   ego_heading = ego_traj_points[:, 2], 
                                                                                   obj_pos = obj_traj_points[:,:2], 
                                                                                   degree= HIST_DEGREE, 
                                                                                   prior_data= prior_data)
    
    for i, (timestamp, ego_traj_point, obj_traj_point) in enumerate(zip(timestamps, ego_traj_points, obj_traj_points)):        
        ego_pos = ego_traj_point[:2]
        ego_psi = ego_traj_point[2]
        obj_pos = obj_traj_point[:2]
        obj_vel = obj_traj_point[-2:]      
        obj_heading = obj_traj_point[2]
        
        vel_obs_cov = np.diag([3, 3])**2
        
        if obj_state is not None:
            # Prediction Step
            dt = timestamp - last_timestamp

            MM = TrajectoryMotion(TRAJ, Q, Prior=motion_prior)

            predicted_obj_state = obj_state.predict(MM, dt)

            # Update Step
            pos_obs_cov = pos_obs_cov_temp[i] #tracking_utils.build_obj_observation_noise_covariance(ego_pos, ego_psi, obj_pos, degree= HIST_DEGREE, prior_data= prior_data)
             
            if i == obj_traj_points.shape[0] -1: #assume less observation noise for the last point
                pos_obs_cov = pos_obs_cov / 9.
                
            OM = TrajectoryObservation(TRAJ, t=HIST_TIMESCALE, derivatives=[[0,1]], R=[block_diag(pos_obs_cov, vel_obs_cov)])
            obj_state = predicted_obj_state.update(np.concatenate([obj_pos, obj_vel]), OM)
        else:
            obj_state = tracking_utils.get_initial_state_agt(obj_pos, obj_vel, HISTORYPRIOR, TRAJ, BASIS, HIST_TIMESCALE)
        
        last_timestamp = timestamp
    
    return np.array(obj_state.x).reshape(-1,2),  np.array(obj_state.P)


def process_argo2_data_with_scenario_parquet(src_file, 
                                             output_path=None, 
                                             process_track = True,
                                             process_map = True):
    scenario_file = glob.glob(os.path.join(src_file, '*.parquet*'))[0]
    map_file = glob.glob(os.path.join(src_file, '*.json*'))[0]
    
    bool_test_data= 'test' in src_file

    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_file)
    mi = map_utils.MapInterpreter(src_file, path_degree = PATH_DEGREE)
    
    file_infos = {
        'scenario_id': scenario.scenario_id,
        'map_id': scenario.map_id,
        'focal_track_id': scenario.focal_track_id,
        'slice_id': scenario.slice_id,
        'city_name': scenario.city_name}
    
    if output_path is not None:
        os.makedirs(os.path.join(output_path, f'{scenario.scenario_id}'), exist_ok=True)
    
    timestamps = scenario.timestamps_ns/ 1e9
    timestamps = timestamps - timestamps[0] # shape (110)
    
    df = pd.read_parquet(scenario_file)
    historical_df = df[df['timestep'] < HIST_LEN]
    timesteps = list(np.sort(df['timestep'].unique()))

    actor_ids = list(historical_df['track_id'].unique())
    actor_ids = list(filter(lambda actor_id: np.sum(historical_df[historical_df['track_id'] == actor_id]['observed'])>=MIN_OBS_LEN, actor_ids))
    historical_df = historical_df[historical_df['track_id'].isin(actor_ids)]
    df = df[df['track_id'].isin(actor_ids)]
    
    # DataFrame for AV ang Agent
    av_df = df[df['track_id'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['track_id'])
    agt_df = df[df['track_id'] == scenario.focal_track_id].iloc
    agent_index = actor_ids.index(agt_df[0]['track_id'])
    
    
    num_actors = len(actor_ids)
    timestep_mask = np.zeros((num_actors, 110), dtype=bool) # booleans indicate if object is observed at each timestamp
    time_window = np.zeros((num_actors, 2), dtype=float) # start and end timestamps for the control points
    objects_type = np.zeros((num_actors), dtype=int)
    tracks_category = np.zeros((num_actors), dtype=int)
    x = np.zeros((num_actors, 110, 5), dtype=float) # [x, y, heading, vx, vy]
    x_mean = np.zeros((num_actors, HIST_DEGREE+1, SPACEDIM), dtype=float) 
    x_cov = np.zeros((num_actors, (HIST_DEGREE+1) * SPACEDIM, (HIST_DEGREE+1) * SPACEDIM), dtype=float)
    agent_id = [None] * num_actors
    
    # make the scene centered at AGT
    origin = np.array([agt_df[HIST_LEN-1]['position_x'], agt_df[HIST_LEN-1]['position_y']])
    theta = np.array(agt_df[HIST_LEN-1]['heading'])
    rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    R_mat = np.kron(np.eye(HIST_DEGREE+1), rotate_mat)
    
    ego_positions = np.array([av_df[:HIST_LEN]['position_x'].values, av_df[:HIST_LEN]['position_y'].values]).T
    ego_headings = np.array(av_df[:HIST_LEN]['heading'].values)
    ego_velocities = np.array([av_df[:HIST_LEN]['velocity_x'].values, av_df[:HIST_LEN]['velocity_y'].values]).T
    
    ego_traj = np.concatenate([ego_positions, ego_headings[:, None], ego_velocities], axis=1) # This is raw data
    
    obj_trajs = []
    
    if process_track:
        av_last_fit_error = None
        agent_last_fit_error = None

        for actor_id, actor_df in df.groupby('track_id'):
            actor_idx = actor_ids.index(actor_id)
            agent_id[actor_idx] = actor_id
            actor_hist_steps = [timesteps.index(timestep) for timestep in historical_df[historical_df['track_id']==actor_id]['timestep']]
            actor_steps = [timesteps.index(timestep) for timestep in df[df['track_id'] == actor_id]['timestep']]
            timestep_mask[actor_idx, actor_steps] = True


            objects_type[actor_idx] = OBJECT_TYPES.index(actor_df['object_type'].unique()[0])
            tracks_category[actor_idx] = actor_df['object_category'].unique()[0]
            
            positions = np.array([actor_df[:]['position_x'].values, actor_df[:]['position_y'].values]).T
            headings = np.array(actor_df[:]['heading'].values)
            velocities = np.array([actor_df['velocity_x'].values, actor_df['velocity_y'].values]).T
            
            obj_traj = np.concatenate([positions, headings[:, None], velocities], axis=1) # This is raw data
            obj_trajs.append(obj_traj)
            

            x[actor_idx, actor_steps, :2] = (positions - origin) @ rotate_mat
            x[actor_idx, actor_steps, 2] = headings - theta
            x[actor_idx, actor_steps, 3:5] = velocities @ rotate_mat

            T_hist = timestamps[actor_hist_steps]
            time_window[actor_idx] = np.array([np.min(T_hist), np.max(T_hist)])

            if actor_id == 'AV':
                cps_mean, cps_cov = track_ego_trajectory(obj_traj[np.where(np.array(actor_hist_steps) < HIST_LEN)], 
                                                         T_hist)
            else:                
                cps_mean, cps_cov = track_obj_trajectory(obj_traj[np.where(np.array(actor_hist_steps) < HIST_LEN)], 
                                                         ego_traj[actor_hist_steps],  
                                                         T_hist, 
                                                         objects_type[actor_idx])


            x_mean[actor_idx] = (cps_mean - origin) @ rotate_mat
            x_cov[actor_idx] = R_mat.T @ cps_cov @ R_mat

            if actor_id == 'AV':
                av_last_fit_error = np.linalg.norm(x_mean[actor_idx, -1] - x[actor_idx, HIST_LEN -1, :2], axis = -1)
            elif actor_id == scenario.focal_track_id:
                agent_last_fit_error = np.linalg.norm(x_mean[actor_idx, -1] - x[actor_idx, HIST_LEN-1, :2], axis = -1)



        track_infos = {
                    'object_type': objects_type, # [N]
                    'track_category': tracks_category, # [N]
                    'timestamps_seconds': timestamps, # [110]
                    'x': x[:, :HIST_LEN], # [N, 50, 5]
                    'y': None if bool_test_data else x[:, HIST_LEN:], # [N, 60, 5]
                    'cps_mean': x_mean, # [N, 6, 2]
                    'cps_cov': x_cov, # [N, 12, 12]
                    'timestep_x_mask': timestep_mask[:, :50], #[N, 50]
                    'timestep_y_mask': timestep_mask[:, 50:], #[N, 60]
                    'time_window': time_window, # [N, 2]
                    'av_index': av_index,
                    'agent_index': agent_index,
                    'agent_ids': agent_id, # [N]
                    'origin': origin, # [2]
                    'ro_mat': rotate_mat, # [2, 2]
                    'av_fit_error': av_last_fit_error,
                    'agent_fit_error': agent_last_fit_error,
                    'num_objects': len(actor_ids),
                    'HIST_DEG': HIST_DEGREE
                   }
    
        track_infos.update(file_infos)
        track_output_file = os.path.join(output_path, f'{scenario.scenario_id}', f'{scenario.scenario_id}_track_infos.pkl')
        with open(track_output_file, 'wb') as f:
            pickle.dump(track_infos, f)
    
    
    if process_map:
        map_infos, outlier_lane_ids =  mi.get_map_features(origin, rotate_mat, city_map = global_map[scenario.city_name])
        
        if len(outlier_lane_ids) > 0:
            #warnings.warn("Ourlier Lane with {} in Scenario {}".format(outlier_lane_ids, scenario.scenario_id))
            outlier_lane_dict[scenario.scenario_id] = outlier_lane_ids
        
        map_infos.update(file_infos)
        map_output_file = os.path.join(output_path, f'{scenario.scenario_id}', f'{scenario.scenario_id}_map_infos.pkl')
    
        with open(map_output_file, 'wb') as f:
            pickle.dump(map_infos, f)
    
    return [len(actor_ids), 
            len(mi.all_lane_ids),
            len(list(mi.avm.vector_pedestrian_crossings.keys())) * 2
           ]

        
def get_infos_from_data(data_path, 
                        output_path=None, 
                        num_workers=16,
                        process_track = True,
                        process_map = True):
    from functools import partial
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)


    src_files = glob.glob(data_path + "/*/", recursive = True)
    src_files.sort() 
    results, error_files= [], []

    for src_file in tqdm(src_files):
        try:
            result = process_argo2_data_with_scenario_parquet(src_file, output_path, process_track = process_track, process_map = process_map)
            results.append(result)
        except:
            error_files.append(src_file)
            warnings.warn("Error with " + src_file)
            continue
    
    results = np.max(np.array(results, dtype = int), axis = 0)
    
    print("Max Obj {}, Max Lane {}, Max CW {}".format(*results))
    
    return results


def create_infos_from_data(raw_data_path, 
                           output_path, 
                           splits, 
                           num_workers=16, 
                           process_track = True,
                           process_map = True,
                           global_map_path = None):
    if not isinstance(splits, list):
        splits = [splits]
    
    global global_map
    if global_map_path is None:
        try:
            with open('data/global_map_A2.pkl', 'rb') as f: 
                global_map = pickle.load(f) # load processed map for acceleration
            print('Found saved A2 global map')
        except:
            print('No saved A2 global map found, initialize a new one')
            global_map = {'austin': {},
                          'washington-dc': {}, 
                          'pittsburgh': {},
                          'palo-alto': {},
                          'dearborn': {},
                          'miami': {}}
            
    else:
        with open(global_map_path, 'rb') as f: 
            global_map = pickle.load(f) # load processed map for acceleration
        print('Found saved A2 global map')
    
    
    for split in splits:
        print('---------------- Preprcessing: ' + split + ' ----------------')
        data_infos = get_infos_from_data(
            data_path=os.path.join(raw_data_path, split),
            output_path=os.path.join(output_path, split + '_processed'),
            num_workers=num_workers,
            process_track = process_track,
            process_map = process_map
        )
        filename = os.path.join(output_path, 'processed_scenarios_' + split + '_infos.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(data_infos, f)
        
                
        filename = os.path.join(output_path, 'outlier_lane_dict_' + split + '.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(outlier_lane_dict, f)
        

        filename = os.path.join(output_path, 'global_map_A2.pkl') if global_map_path is None else global_map_path
        with open(filename, 'wb') as f:
            pickle.dump(global_map, f)
        
                
            
        print('----------------Argoverse2 info ' + split + ' file is saved to %s----------------' % filename)


