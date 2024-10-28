import os
from tqdm import tqdm
import tensorflow as tf
import pickle
import numpy as np
import scipy.spatial.distance as spydist
from scipy.special import binom
from copy import deepcopy

NUM_SAMPLE_POINTS = 8

def M_matrix(n):
    '''
    A matrix that transfers monomial basis function to berstein basis function
    param n: the order of polynomial
    '''
    M = np.zeros([n+1, n+1], dtype = np.float32)
    for i in range(1, n+2):
        for j in range(1, i+1):
            M[i-1,j-1] = binom(n, j-1) * binom(n+1-j, i-j) * (-1)**((i+j)%2)

    return M
    
    
Phi_B = np.array([np.linspace(0,1,NUM_SAMPLE_POINTS) **k for k in range(3+1)], dtype = np.float32).T @ M_matrix(3) 


def flip_indecies_cov(D):
    assert D%2 == 0
    r, c = [], []
    for d in range(D):
        if d % 2 != 0:
            c_new =  np.arange(0, D-1, 2).tolist()
            c = c + c_new
            r = r+[d]*len(c_new)
        else:
            c_new =  np.arange(1, D, 2).tolist()
            c = c + c_new
            r = r+[d]*len(c_new)

            
    return r, c

def generate_file_list_dataset(path_list):
    '''Generate a dataset for scenario files
    input:
        path_list: root directory for processed scenario files
    output:
        file_list_dataset: element is (track_file, map_file)
    '''
    path_list_track = []
    path_list_map  = []
    for root, _, files in tqdm(os.walk(os.path.abspath(path_list))):
        files.sort()
        for file in files:
            if 'track' in file:
                path_list_track.append(os.path.join(root, file))
            elif 'map' in file:
                path_list_map.append(os.path.join(root, file))
            # else:
            #     print(f"File {file} is neither track nor map.")

    file_list_dataset = tf.data.Dataset.from_tensor_slices((path_list_track, path_list_map))
    
    return file_list_dataset


class DataLoaderAV2(object):
    '''Dataloader which load the track and map files and generate training data.
    '''
    def __init__(self, 
                 batch_size, 
                 split, 
                 homogenizing,
                 dist_neighbor_agent, 
                 dist_neighbor_mapel, 
                 dist_neighbor_agent_mapel,
                 num_max_agents = 50,
                 num_max_mapels = 150,
                 dataset = 'A2',
                 ):
        '''
        batch_size: batch size for training
        split: name dataset containing paths of preprocessed track and map data (train/val/test)
        dist_neighbor_agent: distance threshold between neighboring agents, only neighbors are considered in agent social attention
        dist_neighbor_mapel: distance threshold between neighboring map elements, only neighbors are considered in map self attention
        dist_neighbor_agent_mapel: distance threshold between neighboring agent and lane, only neighbor lanes are considered in agent-map attention
        num_max_agents: max. number of agents considered in one scenario
        num_max_mapels: max. number of mapelements considered in one scenario
        '''
        self.batch_size = batch_size
        self.split = split
        self.file_dataset = generate_file_list_dataset("data/datasets/" + split + "_" + dataset + "_processed")
        self.combined_dataset = None
        self.loaded_dataset = None
        self.dataset = dataset
        self.d_neighbor_agent = dist_neighbor_agent
        self.d_neighbor_mapel = dist_neighbor_mapel
        self.d_neighbor_agent_mapel = dist_neighbor_agent_mapel
        self.num_max_agents = num_max_agents if num_max_agents is not None else 211
        self.num_max_mapels = num_max_mapels if num_max_mapels is not None else 338
        self.homogenizing = homogenizing
        
        if self.dataset=='WO':
            assert self.homogenizing # WO must be homogenized
        
    def _load_data(self, path_track, path_map, flag_flip):
        ''' generate data for training from preprocessed data
        input:
            path_track: absolute path for file with track info in a certain scenario
            path_map: absolute path for file with map info in a certain scenario
            flag_flip (bool): whether flip the scenario along the x axis of the focal agent's local coordinate system
        output:
            dict containing: 
                "y": ground truth in AGENT INDIVIDUELL FRAME
                "input_agents": agent features in FOCAL AGENT FRAME
                "input_map": map features in FOCAL AGENT FRAME
                "mask_neighbor_agent":  mask for neighboring agents, dim [num_max_agents, num_max_agents] # TODO: whether ues np.triu_indices to avoid redundant values
                "mask_neighbor_mapel": mask for neighboring map elements, dim [num_max_mapels, num_max_mapels] # TODO: whether ues np.triu_indices to avoid redundant values
                "mask_neighbor_agent_mapel": mask for neighboring map elements of agents, dim [num_max_agents, num_max_mapels]
                "y_mask_all": mask for the y values of all agents (including focal)
                "y_mask_others": mask for the y values of non-focal agents
                "target_agents": target_agents,
                "origin": the original position of focal agent in GLOBAL FRAME
                "ro_mat": the orientation matrix of focal agent in GLOBAL FRAME
                "agents_origin": agents original positions in FOCAL AGENT FRAME
                "agents_theta": agents headings in FOCAL AGENT FRAME
                "mapels_origin": map elements original positions in FOCAL AGENT FRAME
                "mapels_theta": map elements headings in FOCAL AGENT FRAME
                "mask_focal_agent": mask for focal agent
                "mask_scored_agent": mask for scored agents
                "mask_valid_agent": mask for valid agents
                "agent_ids": agent ids
                "scenario_id": scenario_id
        '''
        # load data:
        with open(path_track, 'rb') as f: track_info = pickle.load(f)
        with open(path_map, 'rb') as f: map_info = pickle.load(f)
        

        NUM_CPS_AGENT = track_info['HIST_DEG'] + 1
        NUM_CPS_XY_AGENT = NUM_CPS_AGENT*2
        NUM_CPS_MAPEL = map_info['PATH_DEG'] + 1
        NUM_CPS_XY_MAPEL = NUM_CPS_MAPEL*2
        LEN_AGENT_INPUT = int(NUM_CPS_XY_AGENT + NUM_CPS_XY_AGENT * (NUM_CPS_XY_AGENT+1)/2 + 3)
                
        if self.homogenizing:
            LEN_MAPEL_INPUT = int(NUM_CPS_XY_MAPEL + 1)
        else:
            LEN_MAPEL_INPUT = int(NUM_CPS_XY_MAPEL * 3 + 4)
        
        #lower_triangle_indices = np.tril_indices(NUM_CPS_XY_AGENT)
        
        # check number of agents and map elements:
        num_agents = track_info['num_objects']
        num_mapels = map_info['num_lanes'] + map_info['num_cws']
        
        
        if flag_flip:
            track_info, map_info = self._flip_data(track_info, map_info)
        
        if num_mapels > self.num_max_mapels:
            num_mapels = self.num_max_mapels
            map_info = self._filter_mapels(track_info['cps_mean'][track_info['agent_index']:track_info['agent_index']+1, -1], map_info)
        if num_agents > self.num_max_agents:
            num_agents = self.num_max_agents
            track_info = self._filter_agents(track_info)
        
        # initiate tensors
        input_agents = np.zeros([num_agents,LEN_AGENT_INPUT], dtype=np.float32)
        input_map = np.zeros([num_mapels,LEN_MAPEL_INPUT], dtype=np.float32)
       
        mask_focal_agent = np.zeros(num_agents, dtype=bool)
        mask_scored_agent = np.zeros(num_agents, dtype=bool)
        
        mask_neighbor_agent = np.zeros([num_agents, num_agents], dtype=bool)
        mask_neighbor_mapel = np.zeros([num_mapels, num_mapels], dtype=bool)
        mask_neighbor_agent_mapel = np.zeros([num_agents, num_mapels], dtype=bool)
        
        if self.dataset == "A2":
            y = np.zeros([num_agents, 60, 2], dtype=np.float32)
            y_mask_others = np.ones([num_agents, 60], dtype=bool)
            y_mask_all = np.ones([num_agents, 60], dtype=bool)
        elif self.dataset == "WO":
            y = np.zeros([num_agents, 41, 2], dtype=np.float32)
            y_mask_others = np.ones([num_agents, 41], dtype=bool)
            y_mask_all = np.ones([num_agents, 41], dtype=bool)
        
        # set flat inputs for agents:
        agents_origin = track_info['x'][:, -1, :2].astype(np.float32) #track_info['cps_mean'][:, -1].astype(np.float32) #[N, 2]
        agents_theta = track_info['x'][:, -1, 2].astype(np.float32) # #[N] #np.arctan2(agents_theta_vec[:, 1], agents_theta_vec[:, 0]) 
        agents_R = np.transpose(np.array([[np.cos(agents_theta), -np.sin(agents_theta)],
                                 [np.sin(agents_theta),  np.cos(agents_theta)]], dtype=np.float32), (2,0,1)) #[N,2,2]
        
        
        input_agents[:,:NUM_CPS_XY_AGENT] = (track_info['cps_mean']).reshape(num_agents, -1)
        input_agents[:,-3:-1] = track_info['time_window']
        input_agents[:,-1] = track_info['object_type']            

        
        # load flat input for map elements
        lanes_origin = map_info['map_lane_boundary_cps'][:, -1, 0].astype(np.float32) #[M, 2]
        lanes_theta_vec = (map_info['map_lane_boundary_cps'][:, -1, 1] - map_info['map_lane_boundary_cps'][:, -1, 0]).astype(np.float32) #[M, 2]
        lanes_theta = np.arctan2(lanes_theta_vec[:, 1], lanes_theta_vec[:, 0]) #[M]
        
        lane_center_cps = map_info['map_lane_boundary_cps'][:, 2] # [M,4,2]
        
        if self.homogenizing:
            input_map[:map_info['num_lanes'], :NUM_CPS_XY_MAPEL] = (map_info['map_lane_boundary_cps'][:, -1, :]).reshape(map_info['num_lanes'], -1) # only use lane center
            input_map[:map_info['num_lanes'], -1] = map_info['map_lane_type']
        else:
            input_map[:map_info['num_lanes'], :(NUM_CPS_XY_MAPEL*3)] = (map_info['map_lane_boundary_cps'][:, :, :]).reshape(map_info['num_lanes'], -1)
            input_map[:map_info['num_lanes'], -4:-2] = map_info['map_lane_boundary_type']
            input_map[:map_info['num_lanes'], -2] = map_info['map_lane_type']
            input_map[:map_info['num_lanes'], -1] = map_info['map_lane_is_intersection']
        
        if map_info['num_cws'] > 0:
            cws_origin = map_info['map_cw_boundary_cps'][:, -1, 0].astype(np.float32) #[M, 2]
            cws_theta_vec = (map_info['map_cw_boundary_cps'][:, -1, 1] - map_info['map_cw_boundary_cps'][:, -1, 0]).astype(np.float32) #[M, 2]
            cws_theta = np.arctan2(cws_theta_vec[:, 1], cws_theta_vec[:, 0]) #[M]           

            if self.homogenizing:
                input_map[map_info['num_lanes']:num_mapels, :NUM_CPS_XY_MAPEL] = (map_info['map_cw_boundary_cps'][:, -1, :]).reshape(map_info['num_cws'], -1)
                input_map[map_info['num_lanes']:num_mapels, -1] = map_info['map_cw_type']
            else:
                input_map[map_info['num_lanes']:num_mapels, :(NUM_CPS_XY_MAPEL*3)] = (map_info['map_cw_boundary_cps'][:, :, :]).reshape(map_info['num_cws'], -1)
                input_map[map_info['num_lanes']:num_mapels, -4:-2] = map_info['map_cw_boundary_type']
                input_map[map_info['num_lanes']:num_mapels, -2] = map_info['map_cw_type']
                input_map[map_info['num_lanes']:num_mapels, -1] = 2.0 # padding value for cross walk's "is_intersection" feature

            
            lane_center_cps = np.concatenate([lane_center_cps, map_info['map_cw_boundary_cps'][:, 2]],axis=0)
            
            lanes_origin = np.concatenate([lanes_origin, cws_origin], axis = 0)
            lanes_theta = np.concatenate([lanes_theta, cws_theta], axis = 0)
            
        
        # assign mask for focal and scored agents
        mask_focal_agent[track_info['agent_index']] = True
        mask_scored_agent[track_info['track_category'] >=2] = True

        # calculate mask_neighbor
        center_agents_cps = track_info['cps_mean'][:, -1] #[A, 2]
        center_mapels_cps = np.mean(lane_center_cps, axis=-2)
        
        center_mapels_samples = Phi_B[None, :, :] @ lane_center_cps #[M, 8, 2]
        
        dist_agents = spydist.cdist(center_agents_cps, center_agents_cps)
        dist_mapels = spydist.cdist(center_mapels_cps, center_mapels_cps)
        dist_agent_mapels = np.array([spydist.cdist(center_agents_cps, center_mapels_samples[:, i]) for i in range(NUM_SAMPLE_POINTS)])
        
        
        mask_x = track_info['timestep_x_mask'][:, [-1]] #[A, 1]
        mask_x = mask_x.T & mask_x

        mask_valid_agent = track_info['timestep_x_mask'][:, -1] #[A]
        
        if self.split != 'test':
            mask_y = np.sum(track_info['timestep_y_mask'], axis=-1) # [A]
            mask_y = mask_y>0
            mask_valid_agent = mask_valid_agent & mask_y
        
        mask_neighbor_agent[:num_agents, :num_agents] = dist_agents < self.d_neighbor_agent

        mask_neighbor_agent = mask_neighbor_agent & mask_x

        mask_neighbor_mapel[:num_mapels, :num_mapels] = dist_mapels < self.d_neighbor_mapel
        for i in range(NUM_SAMPLE_POINTS):
            mask_neighbor_agent_mapel[:num_agents, :num_mapels] = mask_neighbor_agent_mapel[:num_agents, :num_mapels] | (dist_agent_mapels[i] < self.d_neighbor_agent_mapel)
        
        # load future trajectory for agents
        if track_info['y'] is not None:
            y[:num_agents] = (track_info['y'][:, :, :2] - agents_origin[:, None, :]) @ agents_R # x,y positions
            
            y_mask_all[~track_info['timestep_x_mask'][:, -1], :] = False
            y_mask_all = y_mask_all & track_info['timestep_y_mask']
            
            y_mask_others = deepcopy(y_mask_all)
            y_mask_others[track_info['agent_index'], :] = False
        
       
        timestep_x_mask = track_info['timestep_x_mask'][:, :49] & track_info['timestep_x_mask'][:, 1:]
        timestep_x_mask = np.repeat(timestep_x_mask[:, :, None], axis=-1, repeats=2)        
        
        origin = tf.cast(track_info['origin'], tf.float32)
        ro_mat = tf.cast(track_info['ro_mat'], tf.float32)
        scenario_id = track_info['scenario_id']
        
      
        return (y,
            input_agents,
            input_map,             # [M, D_M]
            mask_neighbor_agent,   # [A, A]
            mask_neighbor_mapel,   # [M, M]
            mask_neighbor_agent_mapel,  # [A, M]
            y_mask_all,   # [A, 60]
            y_mask_others,   # [A, 60]
            np.array([track_info['agent_index'],], dtype=np.int32),   # [1]
            origin,   # [2]
            ro_mat,   # [2, 2]
            agents_origin,  # [A, 2]
            agents_theta,   # [A]
            lanes_origin,   # [M, 2]
            lanes_theta,    # [M]
            mask_focal_agent,   # [A]
            mask_scored_agent,  # [A]
            mask_valid_agent,   # [A]
            track_info['agent_ids'],    # [A]
            np.array([scenario_id], dtype = str))  # [1]
 
    
    def _create_dict(self, 
                     y,  # [A, 60, 2]
                     input_agents,  # [A, D_A]
                     input_map, # [M, D_M]
                     mask_neighbor_agent, # [A, A]
                     mask_neighbor_mapel, # [M, M]
                     mask_neighbor_agent_mapel, # [A, M]
                     y_mask_all, # [A, 60]
                     y_mask_others, # [A, 60]
                     target_agents, # [1]
                     origin, # [2]
                     ro_mat, # [2, 2]
                     agents_origin, # [A, 2]
                     agents_theta, # [A]
                     lanes_origin, # [M, 2]
                     lanes_theta,  # [M]
                     mask_focal_agent, # [A]
                     mask_scored_agent, # [A]
                     mask_valid_agent, # [A]
                     id_info, # [A]
                     scenario_id): # [1]
        return {"y": y,
                "input_agents": input_agents,
                "input_map": input_map,
                "mask_neighbor_agent": mask_neighbor_agent,
                "mask_neighbor_mapel": mask_neighbor_mapel,
                "mask_neighbor_agent_mapel": mask_neighbor_agent_mapel,
                "y_mask_all": y_mask_all,
                "y_mask_others": y_mask_others,
                "target_agents": target_agents,
                "origin": origin,
                "ro_mat": ro_mat,
                "agents_origin": agents_origin,
                "agents_theta": agents_theta,
                "mapels_origin": lanes_origin,
                "mapels_theta": lanes_theta,
                "mask_focal_agent": mask_focal_agent,
                "mask_scored_agent": mask_scored_agent,
                "mask_valid_agent": mask_valid_agent,
                "agent_ids": id_info,
                "scenario_id": scenario_id
               }
    
        
    def load_process(self, shuffle = False, flip = False):
        if flip:
            flag_flip_dataset = tf.data.Dataset.from_tensor_slices(tf.concat((tf.zeros([self.file_dataset.__len__(),], dtype=bool), tf.ones([self.file_dataset.__len__(),], dtype=bool)), axis=-1))
            self.file_dataset = self.file_dataset.repeat(2)
        else:
            flag_flip_dataset = tf.data.Dataset.from_tensor_slices(tf.zeros([self.file_dataset.__len__()], dtype=bool))
        self.combined_dataset = tf.data.Dataset.zip((self.file_dataset, flag_flip_dataset))
        
        # Shuffle data and create batches
        if shuffle:
            self.combined_dataset = self.combined_dataset.shuffle(buffer_size=self.combined_dataset.__len__())

        self.loaded_dataset = (self.combined_dataset.map(lambda path_data, flag_flip: tf.numpy_function(self._load_data, (path_data[0], path_data[1], flag_flip), 
                                    [tf.float32, tf.float32, tf.float32, bool, bool, bool, bool, bool, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, bool, bool, bool, tf.string, tf.string]), num_parallel_calls=tf.data.AUTOTUNE))

        self.loaded_dataset = self.loaded_dataset.cache()
        
        # Pad after cache to use less memory
        self.loaded_dataset = self.loaded_dataset.padded_batch(self.batch_size, padded_shapes = ([self.num_max_agents, None, None], # y
                                                                                                 [self.num_max_agents, None], # input_agents
                                                                                                 [self.num_max_mapels, None], # input_map, 
                                                                                                 [self.num_max_agents, self.num_max_agents], # mask_neighbour_agent
                                                                                                 [self.num_max_mapels, self.num_max_mapels], # mask_neighbor_mapel
                                                                                                 [self.num_max_agents, self.num_max_mapels], # mask_neighbor_agent_mapel
                                                                                                 [self.num_max_agents, None], # y_mask_all,
                                                                                                 [self.num_max_agents, None], # y_mask_others,
                                                                                                 [None], # target_agents
                                                                                                 [None], # origin
                                                                                                 [None, None], # ro_mat
                                                                                                 [self.num_max_agents, None], # agents_origin
                                                                                                 [self.num_max_agents], # agents_theta
                                                                                                 [self.num_max_mapels, None], # mapels_origin
                                                                                                 [self.num_max_mapels], # mapels_theta
                                                                                                 [self.num_max_agents], # mask_focal_agent
                                                                                                 [self.num_max_agents], # mask_scored_agent
                                                                                                 [self.num_max_agents], # mask_valid_agent
                                                                                                 [self.num_max_agents], # agent_ids
                                                                                                 [None])).map(self._create_dict, num_parallel_calls=tf.data.AUTOTUNE) #scenario_id

        # Make dataset fetch batches in the background during the training of the model.
        self.loaded_dataset = self.loaded_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        
           
    def _filter_agents(self, track_info):
        inds_scored_agents = np.where(track_info['track_category'] >=2)[0]
        inds_special_agents = np.unique(np.append(inds_scored_agents, track_info['av_index']))
        dist_focal_agents = spydist.cdist(track_info['cps_mean'][track_info['agent_index']:track_info['agent_index']+1, -1], track_info['cps_mean'][:, -1])
        dist_focal_agents = np.squeeze(dist_focal_agents, axis=0)
        inds_sorted_dist_focal_agents = np.argsort(dist_focal_agents)
        inds_special_far = np.intersect1d(inds_special_agents, inds_sorted_dist_focal_agents[self.num_max_agents:])
        if inds_special_far.shape[0] > 0:
            inds_unscored_near = np.setdiff1d(inds_sorted_dist_focal_agents[:self.num_max_agents], inds_special_agents, assume_unique=True)
            inds_unscored_near = inds_unscored_near[:-(inds_special_far.shape[0])]
            inds_filtered_agents = np.concatenate((inds_special_agents, inds_unscored_near))#.sort()
        else:
            inds_filtered_agents = inds_sorted_dist_focal_agents[:self.num_max_agents]#.sort()
        
        
        for key in ['object_type', 'track_category', 'x', 'y', 'cps_mean', 'cps_cov', 'timestep_x_mask', 'timestep_y_mask', 'time_window', 'agent_ids']:
            if key == 'agent_ids' and self.dataset!='A2':
                continue
            if self.split == 'test':
                if key == 'y' or key == 'timestep_y_mask':
                    continue
            if key == 'agent_ids':
                track_info[key] = np.array(track_info[key])[inds_filtered_agents]
            else:
                track_info[key] = track_info[key][inds_filtered_agents]

        track_info['av_index'] = (inds_filtered_agents==track_info['av_index']).nonzero()[0].item()
        track_info['agent_index'] = (inds_filtered_agents==track_info['agent_index']).nonzero()[0].item()
        track_info['num_objects'] = self.num_max_agents
        
        return track_info

    
    def _filter_mapels(self, focal_position, map_info):
        lane_center_cps = map_info['map_lane_boundary_cps'][:, 2] # [M,4,2]
        if map_info['num_cws'] > 0:
            lane_center_cps = np.concatenate([lane_center_cps, map_info['map_cw_boundary_cps'][:, 2]],axis=0)
        center_mapels_cps = np.mean(lane_center_cps, axis=-2) # [M,2]
        dist_focal_mapels = spydist.cdist(focal_position, center_mapels_cps)
        dist_focal_mapels = np.squeeze(dist_focal_mapels)
        inds_sorted_dist_focal_mapels = np.argsort(dist_focal_mapels)
        inds_filtered_mapels = inds_sorted_dist_focal_mapels[:self.num_max_mapels]
        inds_filtered_lanes = inds_filtered_mapels[inds_filtered_mapels < map_info['num_lanes']]
        inds_filtered_cws = inds_filtered_mapels[inds_filtered_mapels >= map_info['num_lanes']]
        if inds_filtered_cws.shape[0] > 0:
            inds_filtered_cws -= map_info['num_lanes']
        for key in ['map_lane_boundary_cps', 'map_lane_boundary_type', 'map_lane_type', 'map_lane_is_intersection']:
            map_info[key] = map_info[key][inds_filtered_lanes]
        for key in ['map_cw_boundary_cps', 'map_cw_boundary_type', 'map_cw_type']:
            map_info[key] = map_info[key][inds_filtered_cws]
        map_info['num_lanes'] = inds_filtered_lanes.shape[0]
        map_info['num_cws']  = inds_filtered_cws.shape[0]
        return map_info
    
    
    
    def _flip_data(self, track_infos, map_infos):
        ''' augment dataset by flipping along x-axis of local system
        '''
        # flip track
        track_infos['x'][:, :, [1,2,4]] *= -1. # y, theta, vy
        track_infos['y'][:, :, [1,2,4]] *= -1. # y, theta, vy
        track_infos['cps_mean'][:, :, -1] *= -1.
        r, c = flip_indecies_cov((track_infos['HIST_DEG']+1)*2)
        track_infos['cps_cov'][:, r, c] *= -1.
        track_infos['origin'][-1] *= -1.
        track_infos['ro_mat'][0,1] *= -1.
        track_infos['ro_mat'][1,0] *= -1.
        
        # flip map
        map_infos['map_lane_boundary_cps'][:, :, :, -1] *= -1 #[M, 3, 4, 2]
        map_infos['map_cw_boundary_cps'][:, :, :, -1] *= -1 #[M, 3, 4, 2]
        map_infos['origin'][-1] *= -1.
        map_infos['ro_mat'][0,1] *= -1.
        map_infos['ro_mat'][1,0] *= -1. 
        
        return track_infos, map_infos
        
    def get_batch(self):
        return next(iter(self.loaded_dataset))
