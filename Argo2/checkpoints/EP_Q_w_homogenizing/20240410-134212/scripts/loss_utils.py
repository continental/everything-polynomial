import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

################################################# Regression Loss #########################################################################


################## Euclidean Distance ######################
class minFDE(tf.keras.losses.Loss):
    def __init__(self, top_k=1):
        super(minFDE, self).__init__()
        self.name = 'minFDE'
        self.top_k = top_k
        
    def transform_mask(self, mask):
        return tf.where(mask, tf.ones(mask.shape), tf.zeros(mask.shape))

    def call(self, mess, pred, pi, y_mask = None, length=60, axis = -1,  dtype=tf.float64):
        """
            Parameters:
            -----------
            mess: messurement tf.tensor [B, 60, 2]
                
            pred: prediction tf.tensor [B, K, 60, 2]
            pi: probability tf.tensor [B, K]
            y_mask: tf.tensor [B, K, 60]
        """
        mean = pred
        #assert mess.shape[-2:] == mean.shape[-2:]

        mess = tf.cast(mess[:, :length, :], dtype = dtype)
        mean = tf.cast(mean[:, :, :length, :], dtype = dtype)
        
        prob = tf.nn.softmax(pi, axis=-1)
        top_pi, top_pi_idx  = tf.math.top_k(prob, k=self.top_k, sorted = False)
        mean_top_k = tf.gather(mean, top_pi_idx, batch_dims = 1)
        
        if y_mask == None:
            return tf.reduce_mean(tf.reduce_min(tf.linalg.norm(mean_top_k[:, :, -1, :]-mess[:, None, -1, :], axis=axis), axis = 1))
        else:
            y_mask_f = tf.cast(self.transform_mask(y_mask), dtype = dtype) #[B, K, 60]

            delta_sum = tf.math.multiply(tf.linalg.norm(mean_top_k[:, :, -1, :]-mess[:, None, -1, :], axis=axis), y_mask_f) # [B, K]
            best_mode, best_mode_idx = tf.math.top_k(-delta_sum, k=1, sorted = False)
            best_prediction = tf.gather(mean_top_k, best_mode_idx, batch_dims = 1) # [B, 1, 60,2]
            
            best_delta_mean = tf.linalg.norm(best_prediction[:, 0, -1, :]-mess[:, -1, :], axis=axis) # [B]
            
            y_mask_one_mode = tf.squeeze(tf.where(tf.reshape(y_mask[:, 0, :], [-1])), axis= -1)

            valid_delta_mean = tf.gather(best_delta_mean, y_mask_one_mode, axis = 0, batch_dims=0)
            
            
            return tf.reduce_mean(valid_delta_mean)
    

    
class minADE(tf.keras.losses.Loss):
    def __init__(self, top_k=1):
        super(minADE, self).__init__()
        self.name = 'minADE'
        self.top_k = top_k
    
    def transform_mask(self, mask):
        return tf.where(mask, tf.ones(mask.shape), tf.zeros(mask.shape))
    
    def call(self, mess, pred, pi, y_mask = None, length=60, axis = -1, dtype=tf.float64):
        """
            Parameters:
            -----------
            mess: messurement tf.tensor [B, 60, 2]
                
            pred: prediction tf.tensor [B, K, 60, 2]
            pi: probability tf.tensor [B, K]
            y_mask: tf.tensor [B, K, 60]
        """

        mean = pred
        #assert mess.shape[-2:] == mean.shape[-2:]

        mess = tf.cast(mess[:, :length, :], dtype = dtype)
        mean = tf.cast(mean[:, :, :length, :], dtype = dtype)
        
        
        prob = tf.nn.softmax(pi, axis=-1)
        top_pi, top_pi_idx  = tf.math.top_k(prob, k=self.top_k, sorted = False)
        mean_top_k = tf.gather(mean, top_pi_idx, batch_dims = 1)
        
        if y_mask is None:
            return tf.reduce_mean(tf.reduce_min(tf.reduce_mean(tf.linalg.norm(mean_top_k[:, :, :, :]-mess[:, None, :, :], axis=axis), axis = axis), axis = 1))
        else:
            y_mask_f = tf.cast(self.transform_mask(y_mask), dtype = dtype) #[B, K, 60]
            
            delta_sum = tf.reduce_sum(tf.math.multiply(tf.linalg.norm(mean_top_k[:, :, :, :]-mess[:, None, :, :], axis=axis), y_mask_f), axis = -1) # [B, K]
            best_mode, best_mode_idx = tf.math.top_k(-delta_sum, k=1, sorted = False)
            best_prediction = tf.gather(mean_top_k, best_mode_idx, batch_dims = 1) # [B, 1, 60,2]

            
            best_delta_mean = tf.linalg.norm(best_prediction[:, 0, :, :]-mess[:, :, :], axis=axis) # [B, 60]

            y_mask_one_mode = tf.squeeze(tf.where(tf.reshape(y_mask[:, 0, :], [-1])), axis= -1)

            best_delta_mean = tf.reshape(best_delta_mean, [-1])

            valid_delta_mean = tf.gather(best_delta_mean, y_mask_one_mode, axis = 0, batch_dims=0)

            return tf.reduce_mean(valid_delta_mean)
    
       
class ADE(tf.keras.losses.Loss):
    def __init__(self):
        super(ADE, self).__init__()
        self.name = 'ADE'
        
    def call(self, mess, pred, axis = -1, dtype=tf.float64):
        """
            Parameters:
            -----------
            mess: messurement tf.tensor [B, 2]
                
            pred: prediction tf.tensor [B, 2]
        """
        mess = tf.cast(mess, dtype = dtype)
        pred = tf.cast(pred, dtype = dtype)

        return tf.reduce_mean(tf.linalg.norm(mess-pred, axis=-1))
    
    
################## Negative Log Likelihood ######################

# class NLL_N(tf.keras.losses.Loss):
#     def __init__(self, k = 'traj_mean'):
#         super(NLL_N, self).__init__()
#         self.name = 'NLL_N'
#         self.k = k
        
#     def call(self, mess, pred, dtype=tf.float64):
#         """
#             Parameters:
#             -----------
#             mess: tf.tensor [B, 60, 2]
#                 messurement mean

#             pred: dict 
#                 element 'traj_mean': mean with tf.tensor [B, K, 60, 2]
#                 element 'traj_scale': mean with tf.tensor [B, K, 60, 2, 2]
#         """
#         pred_mean = pred[self.k]
#         pred_scale = pred['traj_scale']

#         assert mess.shape[-2:] == pred_mean.shape[-2:]
#         assert pred_mean.shape[-1] == pred_scale.shape[-1]

#         mess = tf.cast(mess, dtype = dtype)
#         pred_mean = tf.cast(pred_mean, dtype = dtype)
#         pred_scale = tf.cast(pred_scale, dtype = dtype)

#         mvn = tfd.MultivariateNormalTriL(loc=pred_mean, scale_tril=pred_scale)

#         return tf.reduce_mean(tf.reduce_min(-mvn.log_prob(mess[:, None, :, :]), axis = 1))

    
# class NLL_T(tf.keras.losses.Loss):
#     def __init__(self, k):
#         super(NLL_T, self).__init__()
#         self.name = 'NLL_T'
#         self.k = k
        
#     def NLL_T(self, mess, pred, df = 1., dtype=tf.float64):
#         """
#             Parameters:
#             -----------
#             mess: tf.tensor [B, 60, 2]
#                 messurement mean

#             pred: dict 
#                 element 'traj_mean': mean with tf.tensor [B, K, 60, 2]
#                 element 'traj_scale': mean with tf.tensor [B, K, 60, 2, 2]
#         """
#         pred_mean = pred[self.k]
#         pred_scale = pred['traj_scale']

#         assert mess.shape[-2:] == pred_mean.shape[-2:]
#         assert pred_mean.shape[-1] == pred_scale.shape[-1]

#         mess = tf.cast(mess, dtype = dtype)
#         pred_mean = tf.cast(pred_mean, dtype = dtype)
#         pred_scale = tf.cast(pred_scale, dtype = dtype)

#         mvt =  tfd.MultivariateStudentTLinearOperator(df = df, loc=pred_mean, scale= tf.linalg.LinearOperatorLowerTriangular(pred_scale)) 

#         return tf.reduce_mean(tf.reduce_min(-mvt.log_prob(mess[:, None, :, :]), axis = 1))

    
# class Custom_NLL(tf.keras.losses.Loss):
#     def __init__(self, k):
#         super(custom_NLL, self).__init__()
#         self.name = 'custom_NLL'
#         self.k = k
        
#     def call(self, mess, pred, alpha = 0.7, df = 1., dtype=tf.float64):
#         """
#             Parameters:
#             -----------
#             mess: tf.tensor [B, 60, 2]
#                 messurement mean

#             pred: dict 
#                 element 'traj_mean': mean with tf.tensor [B, K, 60, 2]
#                 element 'traj_scale': mean with tf.tensor [B, K, 60, 2, 2]
#         """
#         pred_mean = pred[self.k]
#         pred_scale = pred['traj_scale']

#         assert mess.shape[-2:] == pred_mean.shape[-2:]
#         assert pred_mean.shape[-1] == pred_scale.shape[-1]

#         mess = tf.cast(mess, dtype = dtype)
#         pred_mean = tf.cast(pred_mean, dtype = dtype)
#         pred_scale = tf.cast(pred_scale, dtype = dtype)

#         mvn = tfd.MultivariateNormalTriL(loc=pred_mean, scale_tril=pred_scale)
#         mvt = tfd.MultivariateStudentTLinearOperator(df = df, loc=pred_mean, scale= tf.linalg.LinearOperatorLowerTriangular(pred_scale))

#         return tf.reduce_mean(tf.reduce_min(tf.concat( [-(1- alpha) * mvn.log_prob(mess[:, None, :, :]), 
#                                                         - alpha * mvt.log_prob(mess[:, None, :, :])], axis = 1), axis=1))



################################################# Classification Loss #########################################################################
# class CrossEntropy(tf.keras.losses.Loss):
#     def __init__(self, k):
#         super(CrossEntropy, self).__init__()
#         self.name = 'CrossEntropy'
#         self.k = k

#     def call(self, mess, pred, dtype=tf.float64, smooth_one_hot= True):
#         """
#         Parameters:
#         -----------
#         mess: tf.tensor [B, 60, 2]
#             messurement mean

#         pred: dict 
#             element 'logits': mean with tf.tensor [B, K]
#             element 'traj_mean': mean with tf.tensor [B, K, 60, 2]
#         """
#         mean = pred[self.k]
#         logits = pred['logits']
#         assert mess.shape[-2:] == mean.shape[-2:]

#         mess = tf.cast(mess, dtype = dtype)
#         mean = tf.cast(mean, dtype = dtype)
#         logits = tf.cast(logits, dtype = dtype)
#         bce = tf.keras.losses.BinaryCrossentropy()

#         diff_mean = tf.reduce_mean(tf.linalg.norm(mean-mess[:, None, :, :], axis = -1), axis = -1)
        
#         if smooth_one_hot:
#             one_hot = tf.one_hot(indices = tf.math.argmin(diff_mean, axis = 1), depth= logits.shape[-1], off_value = 0.05, on_value = 1 - 0.05*logits.shape[-1])
#         else:
#             one_hot = tf.one_hot(indices = tf.math.argmin(diff_mean, axis = 1), depth= logits.shape[-1])

#         return bce(one_hot, tf.nn.softmax(logits, axis = -1))

class Mixture_ADE(tf.keras.losses.Loss):
    def __init__(self):
        super(Mixture_ADE, self).__init__()
        self.name = 'Mixture_ADE'
    
    def transform_mask(self, mask):
        return tf.where(mask, tf.ones(mask.shape), tf.zeros(mask.shape))
    
    def call(self, mess, pred, pi, y_mask = None, length=60, axis = -1, dtype=tf.float64):
        """
            Parameters:
            -----------
            mess: messurement tf.tensor [B, 60, 2]
                
            pred: prediction tf.tensor [B, K, 60, 2]
            pi: probability tf.tensor [B, K]
            y_mask: tf.tensor [B, K, 60]
        """

        mean = pred
        prob = tf.nn.softmax(pi, axis=-1)

        #assert mess.shape[-2:] == mean.shape[-2:]
        assert prob.shape[1] == mean.shape[1]

        mess = tf.cast(mess[:, :length, :], dtype = dtype)
        mean = tf.cast(mean[:, :, :length, :], dtype = dtype)
        prob = tf.cast(prob, dtype = dtype)
        
        if y_mask == None:
            return tf.reduce_mean(prob * tf.reduce_mean(tf.linalg.norm(mean[:, :, :, :]-mess[:, None, :, :], axis=axis), axis = axis))
        else:
            y_mask_f = tf.cast(self.transform_mask(y_mask), dtype = dtype) #[B, K, 60]
            delta_mean = tf.repeat(prob[..., None], mean.shape[-2], axis=-1) * tf.math.multiply(tf.linalg.norm(mean[:, :, :, :]-mess[:, None, :, :], axis=axis), y_mask_f) # [B,K, 60]
            y_mask = tf.squeeze(tf.where(tf.reshape(y_mask, [-1])), axis= -1)

            delta_mean = tf.reshape(delta_mean, [-1])

            valid_delta_mean = tf.gather(delta_mean, y_mask, axis = 0, batch_dims=0)

            return tf.reduce_mean(valid_delta_mean)

    
# class Mixture_NLL(tf.keras.losses.Loss):
#     def __init__(self, k):
#         super(CrossEntropy_NLL, self).__init__()
#         self.name = 'CrossEntropy_NLL'
#         self.k

#     def call(self, mess, pred, dtype=tf.float64):
#         """
#         Parameters:
#         -----------
#         mess: tf.tensor [B, 60, 2]
#             messurement mean

#         pred: dict 
#             element 'logits': mean with tf.tensor [B, K]
#             element 'traj_mean': mean with tf.tensor [B, K, 60, 2]
#             element 'traj_scale': mean with tf.tensor [B, K, 60, 2, 2]
#         """
#         pred_mean = pred[self.k]
#         pred_scale = pred['traj_scale']
#         logits = pred['logits']

#         assert mess.shape[-2:] == pred_mean.shape[-2:]
#         assert pred_mean.shape[-1] == pred_scale.shape[-1]
    
#         mess = tf.cast(mess, dtype = dtype)
#         pred_mean = tf.cast(pred_mean, dtype = dtype)
#         pred_scale = tf.cast(pred_scale, dtype = dtype)
#         logits = tf.cast(logits, dtype = dtype)
    
#         mvn = tfd.MultivariateNormalTriL(loc=pred_mean, scale_tril=pred_scale)
#         mvt = tfd.MultivariateStudentTLinearOperator(df = df, loc=pred_mean, scale= tf.linalg.LinearOperatorLowerTriangular(pred_scale))
        
#         ll_n =  tf.reduce_mean(mvn.log_prob(mess[:, None, :, :]), axis=-1) #[B,K]
#         ll_t =  tf.reduce_mean(mvt.log_prob(mess[:, None, :, :]), axis=-1) #[B,K]      
        
#         ll = tf.reduce_min(tf.concat([ll_n[:, None, :], ll_t[:, None, :]], axis=1), axis=1) #[B,K]
        
#         log_pi = tf.nn.log_softmax(logits, axis = -1) # [B, K]
        
#         return tf.math.reduce_logsumexp(log_pi+ll, axis=1)