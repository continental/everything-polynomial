from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model.ep import EP
import utils_common.dataloader_utils as dataloader_utils
import tensorflow as tf
tf.random.set_seed(2024)

DIST_NEIGHBOR_AGENT = 150
DIST_NEIGHBOR_MAPEL = 200
DIST_NEIGHBOR_AGENT_MAPEL = 200

homogenizing= True
flip = False

# Model
model_name = 'EP_F'
hist_const = ([0], [[0]]) # two constraints at 0th timestep, each with derivative 0
pred_const = ([30, 60], [[0,1,2], [0,1,2]]) # three constraints at 30th timestep, each with derivative 0, 1, 2. three constraints at 60th timestep, each with derivative 0, 1, 2
HIST_DEG = 5
PRED_DEG = 6
EMBED_DIM = 64

if model_name == 'EP_F':
    init_lr = 1e-3
    epochs = 128
    BATCH_SIZE = 128 if flip else 64 
elif model_name == 'EP_Q':
    init_lr = 5e-4
    epochs = 64
    BATCH_SIZE = 32
    


if __name__=="__main__":
    split = "train"
    
    # Initiate Dataloaders
    train_dataloader  = dataloader_utils.DataLoaderAV2(batch_size=BATCH_SIZE, 
                                                       split=split, 
                                                       dist_neighbor_agent=DIST_NEIGHBOR_AGENT, 
                                                       dist_neighbor_mapel=DIST_NEIGHBOR_MAPEL, 
                                                       dist_neighbor_agent_mapel=DIST_NEIGHBOR_AGENT_MAPEL,
                                                       dataset = 'A2',
                                                       homogenizing=homogenizing)
    train_dataloader.load_process(shuffle=True, flip = flip)
    train_dataset = train_dataloader.loaded_dataset

    split = "val"
    val_dataloader  = dataloader_utils.DataLoaderAV2(batch_size=BATCH_SIZE, 
                                                     split=split, 
                                                     dist_neighbor_agent=DIST_NEIGHBOR_AGENT, 
                                                     dist_neighbor_mapel=DIST_NEIGHBOR_MAPEL, 
                                                     dist_neighbor_agent_mapel=DIST_NEIGHBOR_AGENT_MAPEL,
                                                     dataset = 'A2',
                                                     homogenizing=homogenizing)
    val_dataloader.load_process(shuffle=False)
    val_dataset = val_dataloader.loaded_dataset

    print("Num. of batches Train: {} \nNum. of batches Val: {}".format(train_dataset.__len__(), val_dataset.__len__()))



    model = EP(pred_deg=PRED_DEG,
               hist_deg=HIST_DEG,
               d_hidden= EMBED_DIM,
               hist_constraints=hist_const,
               pred_constraints=pred_const,
               init_lr = init_lr,
               model_name = model_name,
               homogenizing=homogenizing,
               n_heads = 4,
               n_layers = 1,
               n_modes = 6,
               dropout=0.2,
               return_scale = False)

    # Train Model
    model.train_dataset(train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    epochs=epochs,
                    target = 'focal',
                    model_name = model_name+'')
    
    model.model.summary()