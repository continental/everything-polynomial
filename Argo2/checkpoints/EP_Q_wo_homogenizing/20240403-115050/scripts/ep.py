import tensorflow as tf
from scipy.special import binom
from utils import loss_utils
from utils import utils
from utils.tracking_utils import get_monomial_prior
import tensorflow_probability as tfp
import utils.loss_utils as loss_utils
import model.model as model
import numpy as np
from tqdm.auto import tqdm
import shutil
import os
import datetime
import json
from copy import deepcopy

os.environ["TZ"] = "UTC"

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warmup_steps=1):
        super(WarmupCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.alpha = alpha
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        

    def __call__(self, step):
        # Cosine decay
        cosine_decay  = 0.5 * (1 + tf.cos(tf.cast(np.pi, tf.float64) * tf.cast((step - self.warmup_steps), tf.float64) / tf.cast(self.decay_steps-self.warmup_steps, tf.float64)))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        learning_rate = tf.cast(self.initial_learning_rate*decayed, tf.float64)
        # Target LR * progress of warmup (=1 at the final warmup step)
        warmup_lr = tf.cast(self.initial_learning_rate * (step / self.warmup_steps), tf.float64)

        learning_rate = tf.where(step < self.warmup_steps, warmup_lr, learning_rate)
    
        return learning_rate

class EP(tf.keras.Model):
    def __init__(self,
                 hist_constraints: tuple, # e.g. ([future_step_1, future_step_2, ..., future_step_N], [[derivative_0, derivative_1], [derivative_0, derivative_1], ... [derivative_0, derivative_1]])
                 pred_constraints: tuple, # e.g. ([future_step_1, future_step_2, ..., future_step_N], [[derivative_0, derivative_1], [derivative_0, derivative_1], ... [derivative_0, derivative_1]])
                 d_hidden: int,
                 model_name: str,
                 init_lr: float,
                 homogenizing: bool,
                 pred_deg: int = 5,
                 hist_deg: int = 5,
                 path_deg: int = 3,
                 space_dim: int = 2,
                 pred_steps: int = 60, 
                 hist_steps: int = 50,
                 n_modes: int = 6,
                 return_scale: bool = False,
                 dropout: float = 0.0,
                 dtype = np.float32,
                 alpha = 0.,
                 n_heads = 4,
                 n_layers = 2,
                 key_dim = 16,
                 ) -> None:
        super(EP, self).__init__()
        '''
        params:
            hist_constraints: tuple for describing history constraints, 
            # e.g. ([future_step_1, future_step_2, ..., future_step_N], [[derivative_0, derivative_1], [derivative_0, derivative_1], ... [derivative_0, derivative_1]])
            
            pred_constraints: tuple for describing history constraints, 
            # e.g. ([future_step_1, future_step_2, ..., future_step_N], [[derivative_0, derivative_1], [derivative_0, derivative_1], ... [derivative_0, derivative_1]])
            
            d_hidden: hidden dimension
            pred_deg: polynomial degree of prediction
            hist_deg: polynomial degree of history trajectory
            path_deg: polynomial degree of lane segment
            space_dim: space dimension
            pred_steps: timesteps of prediction 
            hist_steps: timesteps of history
            n_modes: number of modes for predictions
            return_scale: if scale of distribution will be returned
            dropout: dropout rate
        '''
        self.HIST_DEG = hist_deg
        self.PATH_DEG = path_deg
        self.PRED_DEG = pred_deg
        self.HIST_CONST = hist_constraints,
        self.PRED_CONST = pred_constraints,
        self.NUM_HIST_CONST = len([k for ks in hist_constraints[1] for k in ks])
        self.NUM_PRED_CONST = len([k for ks in pred_constraints[1] for k in ks])
        self.PRED_STEPS = pred_steps
        self.HIST_STEPS = hist_steps
        self.NUM_MODES = n_modes
        self.SPACE_DIM = space_dim
        self.KEY_DIM = key_dim
        self.NUM_MODES = n_modes
        self.RETURN_SCALE = return_scale
        self.data_type = dtype
        self.alpha = alpha
        self.lower_triangle_indices = np.tril_indices((self.HIST_DEG+1)*self.SPACE_DIM)
        self.bj = tfp.bijectors.FillScaleTriL(diag_bijector=None, diag_shift=None)
        self.model_name = model_name
        self.init_lr = init_lr
        self.homogenizing = homogenizing

        assert self.NUM_HIST_CONST + self.NUM_PRED_CONST >= self.PRED_DEG+1
        assert (self.HIST_DEG +1) >= self.NUM_HIST_CONST
        assert isinstance(hist_constraints[0], list) and isinstance(hist_constraints[1], list) 
        assert isinstance(pred_constraints[0], list) and isinstance(pred_constraints[1], list)
        assert len(hist_constraints[0]) == len(hist_constraints[1]) == 1 # only consider the last history timestep
        assert len(pred_constraints[0]) == len(pred_constraints[1])
        
        self.hist_constraints = hist_constraints
        self.pred_constraints = pred_constraints
        
        # Prepare Phi_hist
        self.T_hist_ = 4.9
        self.t_hist = tf.linspace(0.0, self.T_hist_, self.HIST_STEPS)
        self.tau_hist = self.t_hist / self.T_hist_
        
        self.M_hist_ = self.M_matrix(self.HIST_DEG)

        
        # Prepare Phi_pred
        self.T_pred_ = 6.0
        self.t_pred = tf.linspace(0.0, self.T_pred_, self.PRED_STEPS+1)
        self.tau_pred = self.t_pred / self.T_pred_
        
        self.M_pred_ = self.M_matrix(self.PRED_DEG)        
        self.Phi_M_pred = np.array([self.t_pred **k for k in range(self.PRED_DEG+1)], dtype = np.float32).T
        self.Phi_B_pred = np.array([self.tau_pred **k for k in range(self.PRED_DEG+1)], dtype = np.float32).T @ self.M_pred_

        self.A_, self.A_inv_ = self.A_matrix()
        self.H_B_hist_list = self.H_matrix_B_list()
        
        self.cls_loss = loss_utils.Mixture_ADE()
        self.reg_loss = loss_utils.minADE(top_k=6)
        self.other_reg_loss = loss_utils.ADE()

        self.val_metrics = [loss_utils.minADE(top_k=6), 
                            loss_utils.minFDE(top_k=6),
                            loss_utils.minADE(top_k=1), 
                            loss_utils.minFDE(top_k=1),
                            ]
        
        self.lr = WarmupCosineDecay(initial_learning_rate = self.init_lr,  decay_steps = 2e5/32*64, alpha= self.alpha, warmup_steps=60000)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)   
        
        self.epoch_train_loss_log, self.epoch_val_loss_log = [], []
        self.epoch_best_train_loss, self.epoch_best_val_loss = np.inf, np.inf
        
        self.model = model.models_dict[model_name](d_hidden=d_hidden, 
                                                   n_pred_const = self.NUM_PRED_CONST, 
                                                   n_heads=n_heads, 
                                                   n_layers=n_layers, 
                                                   homogenizing = self.homogenizing,
                                                   hist_deg = self.HIST_DEG,
                                                   n_modes=self.NUM_MODES, 
                                                   key_dim = self.KEY_DIM, 
                                                   dropout = dropout)
    
    def train_dataset(self, 
                      train_dataset, 
                      val_dataset, 
                      epochs, 
                      target = "focal",
                      model_name= None,
                      log_root_dir = "logs/weights/"):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if model_name == None:
            model_name = self.model_name
        log_dir = os.path.join(log_root_dir, model_name, current_time) 
               
        # Initiate tensorboard summary writer
        train_summary_writer = tf.summary.create_file_writer(log_dir)
        
        ############## Copy training related files ###########################
        if not os.path.exists(log_dir + "/scripts/"):
            os.makedirs(log_dir + "/scripts/")
        
        model_files = os.listdir("model/")
        for file_name in model_files:
            full_file_name = os.path.join("model/", file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, log_dir + "/scripts/")

        shutil.copy("train.py", log_dir + "/scripts/")
        shutil.copy("utils/dataloader_utils.py", log_dir + "/scripts/")
        ######################################################################

        # Start Training
        for epoch in tqdm(range(epochs), desc='epoch'):
            batch_losses = []
            for data in tqdm(train_dataset, desc='train it', leave = False):
                loss, loss_reg, _ = self.train_step(data, training = True, target = target)
                batch_losses.append(loss)

            
            self.epoch_train_loss_log.append(np.mean(batch_losses))
            self.eval_dataset(val_dataset)
            
            # Save the best model
            if self.epoch_val_loss_log[-1][0] < self.epoch_best_val_loss:
                self.epoch_best_val_loss = self.epoch_val_loss_log[-1][0]
        
                checkpoint_path = log_dir + "/best_" + self.reg_loss.name
                self.model.save_weights(checkpoint_path)
                
            with train_summary_writer.as_default():
                tf.summary.scalar('train loss', self.epoch_train_loss_log[-1], step=epoch)
                tf.summary.scalar('val loss', self.epoch_val_loss_log[-1][0], step=epoch)
                
            print('epoch {}: train_loss {}; val_loss {}'.format(epoch, np.round(self.epoch_train_loss_log[-1], 3), np.round(self.epoch_val_loss_log[-1], 3)))

            # Save train and val loss
            self.save_info(log_dir)
    
    def eval_dataset(self, val_dataset, target = "focal"):
        batch_losses = []

        for data in tqdm(val_dataset, desc="val it", leave = False):
            loss = self.eval_step(data, target = target)
            batch_losses.append(loss)
        
        self.epoch_val_loss_log.append(np.mean(batch_losses, axis = 0))
    
    
    @tf.function
    def train_step(self, data, training, target):
        with tf.GradientTape(persistent=True) as tape:
            prediction = self.call(data, training = training)
            
            if self.model_name == 'EP_F':
                y_target, prediction_target = self.get_target(data, prediction, target) # only focal [B*1, 60, 2]
                
                loss_reg_focal = self.reg_loss.call(mess=y_target, pred=prediction_target['traj_mean'], pi=prediction_target['logits'])
                loss_cls_focal = self.cls_loss.call(mess=y_target, pred=prediction_target['traj_mean'], pi=prediction_target['logits'])
                

                y_mask_others = tf.squeeze(tf.reshape(data['y_mask_others'], (-1,1)), axis=-1) # [B*A*60]
                y_others = tf.reshape(data['y'], (-1,2)) # [B*A*60, 2]
                y_hat_others = tf.reshape(prediction['traj_mean_others'], (-1,2)) # [B*A*60, 2]

                y_others = self.filter_batch(y_others, y_mask_others)
                y_hat_others = self.filter_batch(y_hat_others, y_mask_others)

                loss_reg_others = self.other_reg_loss.call(mess=y_others, pred=y_hat_others)
                loss = loss_reg_focal + loss_reg_others + loss_cls_focal
                
            elif self.model_name == 'EP_Q':
                y_mask = tf.repeat(data['y_mask_all'][:, :, None, :], self.NUM_MODES, axis=-2) # [B,A, K, 60]
                y_mask = tf.reshape(y_mask, (-1, self.NUM_MODES, y_mask.shape[-1])) # [B*A, K, 60]
                y = tf.reshape(data['y'], (-1, data['y'].shape[-2], 2)) # [B*A, 60, 2]
                y_hat = tf.reshape(prediction['traj_mean'], (-1, self.NUM_MODES, prediction['traj_mean'].shape[-2],2)) #[B*A, K, 60, 2]
                pi = tf.reshape(prediction['logits'], (-1, self.NUM_MODES)) #[B*A, K]

                loss_reg_all = self.reg_loss.call(mess = y, pred = y_hat, pi = pi, y_mask = y_mask)
                loss_cls_all = self.cls_loss.call(mess = y, pred = y_hat, pi = pi, y_mask = y_mask)
                loss = loss_reg_all + loss_cls_all
            else:
                raise "Unknown Model"
            
        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients((grad, var) 
                    for (grad, var) in zip(grads, self.model.trainable_variables) 
                    if grad is not None)
        
        if self.model_name == 'EP_F':
            return loss, loss_reg_focal, loss_cls_focal
        else:
            return loss, loss_reg_all, loss_cls_all
    
    @tf.function
    def eval_step(self, data, target):
        prediction = self(data, training = False)
        y_target, prediction_target = self.get_target(data, prediction, target)

        loss = [m.call(y_target, prediction_target['traj_mean'], pi = prediction_target['logits']) for m in self.val_metrics]
        
        return loss
        
        
    @tf.function
    def call(self, data, training, origin = None, ro_mat = None) -> tf.Tensor:
        '''
        Params:
            data: batched data
            training: if training mode is activated
            origin: origin position of agent(s) [B, N, 2]
            ro_mat: rotation matrix of agent(s) [B, N, 2, 2]
        '''
        # Output is a tuple of 3 elements. B = Batch, N = #Objects, K = #Predictions
        # logits: [B, N(1), K(6)]
        # mean: [B, N(1), K(6), 6, 2] 
        # N_scale: [B, N(1), K(6), 6, 2, 2] 

        prediction = self.model(data, training = training)
        
        cps_mean_hist = prediction['input_a_cps'] # [B,A,6,2]
        # cps_mean_hist = data['input_agents'][:, :, :(self.HIST_DEG+1)*self.SPACE_DIM]
        # cps_mean_hist = tf.reshape(cps_mean_hist, (cps_mean_hist.shape[0], cps_mean_hist.shape[1], -1, self.SPACE_DIM))

        cps_cov_hist = None
        
        if self.RETURN_SCALE:
            cps_scale_hist = self.bj(target_agent_data[:, :, 12:90])
            cps_cov_hist = cps_scale_hist @ tf.transpose(cps_scale_hist, (0,1,3,2))
 
        pred_traj_mean, pred_traj_scale = self.traj_from_constraints(cps_mean_hist,
                                                                     cps_cov_hist,
                                                                     prediction['y_const_mean'],
                                                                     None,
                                                                     )

        
        if self.model_name == 'EP_F':
            pred_traj_mean_others, _ = self.traj_from_constraints(cps_mean_hist,
                                                                  cps_cov_hist,
                                                                  prediction['y_const_mean_others'],
                                                                  None,
                                                                  )
        
        
        
        
        if origin is not None and ro_mat is not None: 
            # Transform predictions back to original coordinate
            origin_expand = origin[:, None, None, :] 
            ro_mat_expand = ro_mat
            ro_mat_T_expand = tf.transpose(ro_mat_expand, (0,1,3,2))

            pred_traj_mean = tf.einsum('BNKTF, BNFG -> BNKTG', pred_traj_mean, ro_mat_T_expand) + origin_expand
            if self.RETURN_SCALE:
                pred_traj_scale = tf.linalg.cholesky(
                                    ro_mat_expand @ pred_traj_scale @ tf.transpose(pred_traj_scale, (0,1,2,3,5,4)) @ ro_mat_T_expand)
        
        return {
                'logits': prediction['logits'], # [B, N, K]
                'traj_mean': pred_traj_mean, # [B, N, K, 60, 2]
                'traj_scale': pred_traj_scale, # # [B, N, K, 60, 2, 2]
                'traj_mean_others': pred_traj_mean_others if self.model_name == 'EP_F' else None, #[B,N,K,60,2]
               }
    
    
    def traj_from_constraints(self, cps_mean_hist, cps_cov_hist, constraints_mean_pred, constraints_scale_pred):
        '''
        param:
            cps_mean_hist: control points for history trajectory. [B, N, 6, 2]
            cps_cov_hist: covariance of control points for history trajectory. [B, N, 12, 12]
            constraints_mean_pred: constraints for future trajectory. [B, N, K, C_pred, 2]
            constraints_scale_pred: scale of constraints for future trajectory. [B, N, K, C_pred, 2, 2] or [B, N, K, 2*C_pred, 2*C_pred]
            
        output:
            traj_pred_mean: mean of future trajectory points. [B, N, K, 60, 2]
            traj_pred_scale: scale (Gaussian) of future trajectory points. [B, N, K, 60, 2, 2]
        '''
        B, N, K = constraints_mean_pred.shape[:3]
        constraints_cov_pred, constraints_cov = None, None

        # get histrory constraints
        constraints_mean_hist,  constraints_cov_hist = self.get_hist_constraints(cps_mean_hist, cps_cov_hist)

        
        if self.RETURN_SCALE:
            constraints_cov_pred = constraints_scale_pred @ tf.transpose(constraints_scale_pred, (0, 1, 2, 3, 5, 4))
            
            linop_blocks = None
            if constraints_scale_pred.ndim == 6: # time independent points
                linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in 
                               [tf.repeat(constraints_cov_hist[:, :, None, :, :], K, axis = 2)] + [constraints_cov_pred[:, :, :, i] for i in range(self.NUM_PRED_CONST)]]

            else: # time dependent points
                linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in 
                                [tf.repeat(constraints_cov_hist[:, :, None, :, :], K, axis = 2), constraints_cov_pred]]

            
            constraints_cov = tf.linalg.LinearOperatorBlockDiag(linop_blocks).to_dense()
        
        # constraints_mean shape: [B, N, K, 2*C]
        constraints_mean = tf.concat([tf.repeat(constraints_mean_hist[:, :, None, :], K, axis=2),
                                      tf.reshape(constraints_mean_pred, (B, N, K, -1))], axis=-1)
        
        
        params_pred_mean, params_pred_cov =  self.constraints_to_parameters(constraints_mean, constraints_cov)
        
        Phi_M_pred_kron = np.kron(self.Phi_M_pred[1:], np.eye(self.SPACE_DIM, dtype=self.data_type))
        Phi_M_pred_kron_reshape = tf.reshape(Phi_M_pred_kron, (self.PRED_STEPS, self.SPACE_DIM, -1))

        
        traj_pred_mean = tf.reshape(params_pred_mean @ tf.transpose(Phi_M_pred_kron), 
                                    (B, N, K, 60, 2))
        
        traj_pred_scale = None
        
        if self.RETURN_SCALE:
            '''
            T: timesteps 60
            D: space dim 2
            B: batch size
            N: number of objs
            K: number of modes
            F: feature dims
            G: feature dims = F
            '''
            traj_pred_cov = tf.einsum('TDF, BNKFG -> BNKTDG', Phi_M_pred_kron_reshape, params_pred_cov) @ tf.transpose(Phi_M_pred_kron_reshape, (0,2,1))
            traj_pred_scale = tf.linalg.cholesky(traj_pred_cov)

        
        return traj_pred_mean, traj_pred_scale
    
    
    def traj_from_cps(self, cps_mean_hist, cps_cov_hist, cps_mean_pred, cps_scale_pred):
        '''
        param:
            cps_mean_hist: control points for history trajectory. [B, N, 6, 2]
            cps_cov_hist: covariance of control points for history trajectory. [B, N, 12, 12]
            cps_mean_pred: control points for future trajectory. [B, N, K, 6, 2]
            cps_scale_pred: scale (Gaussian) of control points for future trajectory. [B, N, K, 6, 2, 2] or [B, N, K, 12, 12]
            
        output:
            traj_pred_mean: mean of future trajectory points. [B, N, K, 60, 2]
            traj_pred_scale: scale (Gaussian) of future trajectory points. [B, N, K, 60, 2, 2]
        '''
        B, N, K = cps_mean_pred.shape[0], cps_scale_pred.shape[1], self.NUM_MODES
        
        Phi_B_pred_kron = np.kron(self.Phi_B_pred[1:], np.eye(self.SPACE_DIM, dtype=self.data_type))
        Phi_B_pred_kron_reshape = tf.reshape(Phi_B_pred_kron, (self.PRED_STEPS, self.SPACE_DIM, -1))

        
        traj_pred_mean = tf.reshape(cps_mean_pred @ tf.transpose(Phi_B_pred_kron), 
                                    (B, N, K, 60, 2))
        
        traj_pred_scale = None
        
        if self.RETURN_SCALE:
            '''
            T: timesteps 60
            D: space dim 2
            B: batch size
            N: number of objs
            K: number of modes
            F: feature dims
            G: feature dims = F
            '''
            cps_cov_pred = cps_scale_pred @ tf.transpose(cps_scale_pred, (0,1,2,3,5,4) if cps_scale_pred.ndim == 6 else (0,1,2,4,3))
            
            if cps_scale_pred.ndim == 6:
                #transform cps_cov_pred to shape [B, N, K, 12, 12]
                linop_blocks = [tf.linalg.LinearOperatorFullMatrix(cps_cov_pred[:, :, :, c]) for c in range(cps_cov_pred.shape[3])]
                cps_cov_pred = tf.linalg.LinearOperatorBlockDiag(linop_blocks).to_dense() # [B, N, K, 12, 12]
            
            traj_pred_cov = tf.einsum('TDF, BNKFG -> BNKTDG', Phi_B_pred_kron_reshape, cps_cov_pred) @ tf.transpose(Phi_B_pred_kron_reshape, (0,2,1))
            traj_pred_scale = tf.linalg.cholesky(traj_pred_cov)

        
        return traj_pred_mean, traj_pred_scale

    
    
    def get_hist_constraints(self, cps_mean_hist, cps_cov_hist = None):
        '''
        param:
            cps_mean_hist: control points for history trajectory. [B, N, 6, 2]
            cps_cov_hist: covariance of control points for history trajectory. [B, N, 12, 12]
            
        output:
            constraints_mean_hist: constraints for history trajectory. [B, N, C_hist * 2]
            constraints_cov_hist: constraints for history trajectory. [B, N, C_hist * 2, C_hist * 2]
        '''        
        constraints_mean_hist = tf.reshape(cps_mean_hist, (cps_mean_hist.shape[0], cps_mean_hist.shape[1], -1)) @ self.H_B_hist_list.T

        constraints_cov_hist = None
        
        if cps_cov_hist is not None and self.RETURN_SCALE:
            constraints_cov_hist = self.H_B_hist_list @ cps_cov_hist @ self.H_B_hist_list.T
   
        return constraints_mean_hist, constraints_cov_hist
    
    
    def constraints_to_parameters(self, constraints_mean, constraints_cov):
        '''
        param:
            constraints_mean: constraints of polynomial. [B, N, K, 2C]
            constraints_n_cov: covariance of the constraints. [B, N, K, 2C, 2C]
        
        output:
            params_mean: parameters of polynomial (monomial). [B, N, K, 12]
            params_n_cov: covariance of the parameters of polynomial (monomial). [B, N, K, 12, 12]
        '''
        params_mean =  constraints_mean @ self.A_inv_.T
        params_cov = None
        
        if self.RETURN_SCALE:
            params_cov = self.A_inv_ @ constraints_cov @ self.A_inv_.T
            
        return params_mean, params_cov  
    
    
    def constraints_to_parameters_bayesian(self, constraints_mean, constraints_n_cov):
        '''
        not implemented
        
        param:
            constraints_mean: constraints of polynomial. [B, N, K, 2C]
            constraints_n_cov: covariance of the constraints. [B, N, K, 2C, 2C]
        
        output:
            params_mean: parameters of polynomial (monomial). [B, N, K, 12]
            params_cov: covariance of the parameters of polynomial (monomial). [B, N, K, 12, 12]
        
        '''
        print('this function is not implemented')
        pass 
    
    
    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)
    
    #################################### Utils ################################################   
    def D_matrix(self, n, k): 
        '''
        A derivative matrix for monomial basis function
        param n: the order of polynomial
        param k: the order of derivatives
        '''
        D=np.eye(n+1, dtype = self.data_type)
        for i in range(k):
            D = D @ np.diag(np.arange(1,n+1), 1)

        return D
    
    
    def M_matrix(self, n):
        '''
        A matrix that transfers monomial basis function to berstein basis function
        param n: the order of polynomial
        '''
        M = np.zeros([n+1, n+1], dtype = self.data_type)
        for i in range(1, n+2):
            for j in range(1, i+1):
                M[i-1,j-1] = binom(n, j-1) * binom(n+1-j, i-j) * (-1)**((i+j)%2)

        return M
    
    
    def A_matrix(self):
        '''
        A @ parameters = constriants
        build the A matrix upon
        
        return the A matrix [NUM_CONST * SPACE_DIM, (PRED_DEG + 1) * SPACE_DIM]
        '''
        
        A_hist = [self.Phi_M_pred[step] @ self.D_matrix(self.PRED_DEG, k) 
                  for (step, ks) in zip(self.hist_constraints[0], self.hist_constraints[1]) 
                  for k in ks]
        A_pred = [self.Phi_M_pred[step] @ self.D_matrix(self.PRED_DEG, k) 
                  for (step, ks) in zip(self.pred_constraints[0], self.pred_constraints[1]) 
                  for k in ks]

        A = np.vstack(A_hist + A_pred)
        
        A_inv = np.linalg.solve(A.T@A, A.T)

        return np.kron(A, np.eye(self.SPACE_DIM)).astype(self.data_type), np.kron(A_inv, np.eye(self.SPACE_DIM)).astype(self.data_type)
    
    
    def _H_matrix_B_(self, tau, n, d, Delta_t, k):
        '''
        The observation matrix based on control points.
        param tau: scale between [0,1]
        param n: the order of polynomial
        param d: space dimension
        param Delta_t: the time length of trajectory
        param k: the order of derivatives
        
        return a list of observation matricies, each with the shape [self.NUM_HIST_CONST * self.SPACE_DIM, (self.HIST_DEG + 1) * self.SPACE_DIM]
        '''
        time_scale = 1/(Delta_t**k)
        phi_k = self.phi_m(tau, self.HIST_DEG) @ self.D_matrix(self.HIST_DEG, k)
        phi_k_b = phi_k @ self.M_hist_

        return time_scale * np.kron(phi_k_b, np.eye(self.SPACE_DIM))
    
    def H_matrix_B_list(self):
        '''
        create a list of self.NUM_HIST_CONST observation matrices for getting the start states (constraints) from history trajectory
        '''
        H_list = [self._H_matrix_B_(tau = 1.0, # 1.0 for end states
                                 n = self.HIST_DEG, 
                                 d = self.SPACE_DIM, 
                                 Delta_t = self.T_hist_, k = k) for (_, ks) in zip(self.hist_constraints[0], self.hist_constraints[1])
                                 for k in ks]
        
        H_list = np.vstack(H_list)
        
        return H_list.astype(self.data_type)
    
    
    def phi_m(self, tau, n):
        '''
        create monomial basis function
        
        param tau: time
        param n: degree of polynomial
        '''
        return np.array([tau **k for k in range(n+1)], dtype = self.data_type).T
    
    def phi_b(self, tau, n):
        '''
        create bernstein basis fuction
        
        param tau: scaled time between [0,1]
        param n: degree of polynomial
        '''
        return self.phi_m(tau, n) @ self.M_matrix(n)
    
    def get_target(self, data, prediction, target):
        y = tf.reshape(data['y'], [-1] + data['y'].shape[2:].as_list()) # [B*N, 60, 2]
        target_index = tf.squeeze(tf.where(tf.reshape(data['mask_' + target + '_agent'], [-1])), axis=-1) # [B*N]
        y_target = tf.gather(y, target_index, axis = 0, batch_dims=0)
        prediction_target = {}
        for k, v in prediction.items():
            if v is not None:
                prediction_target[k] = tf.gather(tf.reshape(v, [-1]+v.shape[2:].as_list()), target_index, axis = 0, batch_dims=0)
            else:
                prediction_target[k] = None
                
        return y_target, prediction_target
    
    
    def filter_batch(self, data, target_mask):
        assert data.shape[0] == target_mask.shape[0]
        target_index = tf.squeeze(tf.where(target_mask), axis=-1)
        data_filtered = tf.gather(data, target_index, axis = 0, batch_dims=0)
    
        return data_filtered
        
    def save_info(self, log_dir):
        result = {'train_loss': self.epoch_train_loss_log,
                  'val_loss': self.epoch_val_loss_log}
        
        with open(log_dir + '/result.json', "w") as write_file:
            json.dump(result, write_file, cls=utils.NumpyEncoder)