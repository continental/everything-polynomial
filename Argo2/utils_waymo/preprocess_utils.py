'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import sys, os
import numpy as np
import json
import pickle
import tensorflow as tf
import glob
import pandas as pd
from tqdm.auto import tqdm
from waymo_open_dataset.protos import scenario_pb2
from utils_common.helper_utils import waymo_object_type_converter, waymo_protobuf_to_dataframe

sys.path.append('../cmotap/continental-multi-object-tracker-and-predictor/')
from src.cmotap.trajectory import Trajectory 
from src.cmotap.basisfunctions.bernsteinpolynomials import BernsteinPolynomials
from src.cmotap.motionmodels.trajectory_motion import TrajectoryMotion
from src.cmotap.observationmodels.trajectory_observation import TrajectoryObservation

from src.cmotap import utils

import utils_waymo.tracking_utils as tracking_utils 
import utils_waymo.map_utils as map_utils 

from scipy.linalg import block_diag
from pathlib import Path
import time
import warnings


# Load prior parameters
with open('priors_waymo/vehicle/vehicle_5s.json', "r") as read_file:
    prior_vechicle = json.load(read_file)
    
with open('priors_waymo/cyclist/cyclist_5s.json', "r") as read_file:
    prior_cyclist = json.load(read_file)
    
with open('priors_waymo/pedestrian/pedestrian_5s.json', "r") as read_file:
    prior_pedestrian = json.load(read_file)
    
with open('priors_waymo/ego/ego_5s.json', "r") as read_file:
    prior_ego = json.load(read_file)


HIST_TIMESCALE = 4.9
HIST_DEGREE = 5 # According to AIC, best degree for 5s-trajectory of vehicle, cyclist and pedestrian are 5-deg polynomial 
PATH_DEGREE = 3 # Degree for lane segments and cross walks
SPACEDIM = 2
HIST_LEN = 50
MIN_OBS_LEN = 1

BASIS = BernsteinPolynomials(HIST_DEGREE)
TRAJ = Trajectory(basisfunctions=BASIS, spacedim=SPACEDIM, timescale=HIST_TIMESCALE)
    
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
    
    motion_prior = np.eye(BASIS.size * SPACEDIM) * 200000 # uninformative prior for motion #np.kron(np.diag([1e6, 1e6, 1e6, 1e4, 1e3, 1e4]), np.eye(SPACEDIM))  
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
            pos_obs_cov = pos_obs_cov_temp[i]
            
            if i == obj_traj_points.shape[0] -1: #assume less observation noise for the last point
                pos_obs_cov = pos_obs_cov / 9.
                
            OM = TrajectoryObservation(TRAJ, t=HIST_TIMESCALE, derivatives=[[0,1]], R=[block_diag(pos_obs_cov, vel_obs_cov)])
            obj_state = predicted_obj_state.update(np.concatenate([obj_pos, obj_vel]), OM)
        else:
            obj_state = tracking_utils.get_initial_state_agt(obj_pos, obj_vel, HISTORYPRIOR, TRAJ, BASIS, HIST_TIMESCALE)
        
        last_timestamp = timestamp
    
    return np.array(obj_state.x).reshape(-1,2),  np.array(obj_state.P)


def process_waymo_data_with_scenario_proto(data_file, 
                                           output_path=None,
                                           process_track=True,
                                           process_map=True):
    dataset = tf.data.TFRecordDataset(data_file, compression_type='')
    p = Path(data_file)
    file_name = p.name
    
    error_files = {file_name: []}
    
    bool_test_data= 'test' in data_file
    
    for cnt, data in tqdm(enumerate(dataset)):

        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data.numpy()))

        file_infos = {
            'scenario_id': scenario.scenario_id,
            'file_name': file_name,
            #'timestamps_seconds': scenario.timestamps_seconds,
            'current_time_index': scenario.current_time_index,
            'sdc_track_index': scenario.sdc_track_index,
            'objects_of_interest': [obj for obj in scenario.objects_of_interest]}
        
        if output_path is not None:
            os.makedirs(os.path.join(output_path, f'{scenario.scenario_id}'), exist_ok=True)
    
        timestamps = np.array(scenario.timestamps_seconds) # shape (91)
        
       
        # convert to dataframe 
        df_dict, find_focal_agt = waymo_protobuf_to_dataframe(scenario, hist_len = HIST_LEN, min_obs_len = MIN_OBS_LEN)
        
        if not find_focal_agt:
            print('no focal in scenario {}, file {}'.format(scenario.scenario_id, file_name))
            error_files[file_name].append(scenario.scenario_id)
            continue
        
        df = df_dict['df']
        av_df = df_dict['av_df']
        agt_df = df_dict['agt_df']
        historical_df = df_dict['historical_df']
        actor_ids = df_dict['actor_ids']
        av_id = df_dict['av_id']
        focal_track_id = df_dict['focal_track_id']
        scored_track_id = df_dict['scored_track_id']
        av_index = df_dict['av_index']
        agent_index = df_dict['agent_index']
        
        num_actors = len(actor_ids)
        timestep_mask = np.zeros((num_actors, 91), dtype=bool) # booleans indicate if object is observed at each timestamp
        time_window = np.zeros((num_actors, 2), dtype=float) # start and end timestamps for the control points
        objects_type = np.zeros((num_actors), dtype=int)
        tracks_category = np.zeros((num_actors), dtype=int)
        x = np.zeros((num_actors, 91, 5), dtype=float) # [x, y, heading, vx, vy]
        x_mean = np.zeros((num_actors, HIST_DEGREE+1, SPACEDIM), dtype=float) 
        x_cov = np.zeros((num_actors, (HIST_DEGREE+1) * SPACEDIM, (HIST_DEGREE+1) * SPACEDIM), dtype=float)
        
        
        # make the scene centered at AGT
        origin = np.array([agt_df[HIST_LEN-1]['position_x'], agt_df[HIST_LEN-1]['position_y']])
        theta = np.array(agt_df[HIST_LEN-1]['heading'])
        rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
        R_mat = np.kron(np.eye(HIST_DEGREE+1), rotate_mat)

        ego_positions = np.array([[x, y] for (x,y) in zip(av_df[:50]['position_x'].values, av_df[:50]['position_y'].values)])
        ego_headings = np.array([v for v in av_df[:50]['heading'].values])
        ego_velocities = np.array([[vx, vy] for (vx, vy) in zip(av_df[:50]['velocity_x'].values, av_df[:50]['velocity_y'].values)])

        ego_traj = np.concatenate([ego_positions, ego_headings[:, None], ego_velocities], axis=1) # This is raw data
        
        
        if process_track:
            av_last_fit_error = None
            agent_last_fit_error = None

            for actor_id, actor_df in df.groupby('track_id'):
                actor_idx = actor_ids.index(actor_id)
                 
                actor_time_mask = actor_df['valid']  # [91]
                actor_steps = np.where(actor_time_mask==True)[0]
                actor_hist_steps = np.where(actor_time_mask[:HIST_LEN]==True)[0]
                timestep_mask[actor_idx, actor_time_mask] = True


                objects_type[actor_idx] = waymo_object_type_converter(actor_df['object_type'].unique()[0])
                
                if actor_id == focal_track_id: # focal track
                    tracks_category[actor_idx] = 3
                elif actor_id in scored_track_id: # scored track
                    tracks_category[actor_idx] = 2
                elif np.sum(actor_time_mask[:HIST_LEN]) < HIST_LEN: # track_fragment
                    tracks_category[actor_idx] = 0
                else: # unscored track
                    tracks_category[actor_idx] = 1
                    

                positions = np.array([[x, y] for (x,y) in zip(actor_df['position_x'].values, actor_df['position_y'].values)])
                headings = np.array([v for v in actor_df['heading'].values])
                velocities = np.array([[vx, vy] for (vx,vy) in zip(actor_df['velocity_x'].values, actor_df['velocity_y'].values)])
                
                positions = positions[actor_steps]
                headings = headings[actor_steps]
                velocities = velocities[actor_steps]
                
                obj_traj = np.concatenate([positions, headings[:, None], velocities], axis=1) # This is raw data

                x[actor_idx, actor_steps, :2] = (positions - origin) @ rotate_mat
                x[actor_idx, actor_steps, 2] = headings - theta
                x[actor_idx, actor_steps, 3:5] = velocities @ rotate_mat

                T_hist = timestamps[actor_hist_steps]
                time_window[actor_idx] = np.array([np.min(T_hist), np.max(T_hist)])


                if actor_id == av_id:
                    cps_mean, cps_cov = track_ego_trajectory(obj_traj[np.where(np.array(actor_hist_steps) < HIST_LEN)], 
                                                             T_hist)
                else:
                    cps_mean, cps_cov = track_obj_trajectory(obj_traj[np.where(np.array(actor_hist_steps) < HIST_LEN)], 
                                                             ego_traj[actor_hist_steps],  
                                                             T_hist, 
                                                             objects_type[actor_idx])


                x_mean[actor_idx] = (cps_mean - origin) @ rotate_mat
                x_cov[actor_idx] = R_mat.T @ cps_cov @ R_mat

                if actor_id == av_id:
                    av_last_fit_error = np.linalg.norm(x_mean[actor_idx, -1] - x[actor_idx, HIST_LEN -1, :2], axis = -1)
                
                if actor_id == focal_track_id:
                    agent_last_fit_error = np.linalg.norm(x_mean[actor_idx, -1] - x[actor_idx, HIST_LEN-1, :2], axis = -1)
            
            track_infos = {
                        'object_type': objects_type, # [N]
                        'track_category': tracks_category, # [N]
                        'timestamps_seconds': timestamps, # [91]
                        'x': x[:, :HIST_LEN], # [N, 50, 2]
                        'y': None if bool_test_data else x[:, HIST_LEN:], # [N, 41, 2]
                        'cps_mean': x_mean, # [N, 6, 2]
                        'cps_cov': x_cov, # [N, 12, 12]
                        'timestep_x_mask': timestep_mask[:, :50], #[N, 50]
                        'timestep_y_mask': timestep_mask[:, 50:], #[N, 41]
                        'time_window': time_window, # [N, 2]
                        'av_index': av_index,
                        'agent_index': agent_index,
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
            map_infos =  map_utils.get_map_features(scenario, origin, rotate_mat, global_map)

            map_infos.update(file_infos)
            map_output_file = os.path.join(output_path, f'{scenario.scenario_id}', f'{scenario.scenario_id}_map_infos.pkl')

            with open(map_output_file, 'wb') as f:
                pickle.dump(map_infos, f)
            
    return error_files



def get_infos_from_protos(data_path, 
                          output_path=None, 
                          num_workers=16,
                          process_track = True,
                          process_map = True,):

    src_files = glob.glob(os.path.join(data_path, '*.tfrecord*'))
    src_files.sort()
    
    error_logs = {}
    
    for src_file in tqdm(src_files):
        try:
            error_log = process_waymo_data_with_scenario_proto(src_file, output_path, process_track = process_track, process_map = process_map)
            error_logs.update(error_log)
        except:
            warnings.warn("Error with " + src_file)
            continue
    
    
    return error_logs


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
            with open('data/global_map_WO.pkl', 'rb') as f: 
                global_map = pickle.load(f) # load processed map for acceleration
            print('Found saved WO global map')
        except:
            print('No saved WO global map found, initialize a new one')
            global_map = {}
            
    else:
        with open(global_map_path, 'rb') as f: 
            global_map = pickle.load(f) # load processed map for acceleration
        print('Found saved WO global map')
    
    for split in splits:
        print('---------------- Preprcessing: ' + split + ' ----------------')
        error_logs = get_infos_from_protos(
            data_path=os.path.join(raw_data_path, split),
            output_path=os.path.join(output_path, split + '_processed'),
            num_workers=num_workers,
            process_track = process_track,
            process_map = process_map,
        )
        
        filename = os.path.join(output_path, 'error_logs_' + split + '.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(error_logs, f)
            
        
        filename = os.path.join(output_path, 'global_map_WO.pkl') if global_map_path is None else global_map_path
        with open(filename, 'wb') as f:
            pickle.dump(global_map, f)    
        
            
        print('---------------- Finished ----------------')
    
