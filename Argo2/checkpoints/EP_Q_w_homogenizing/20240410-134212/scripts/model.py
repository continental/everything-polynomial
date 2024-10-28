import tensorflow as tf
import tensorflow_probability as tfp
import model.base_module as model_utils


class EP_Base(tf.keras.Model):
    def __init__(self, 
                 d_hidden, 
                 n_pred_const,
                 n_heads, 
                 n_layers, 
                 homogenizing,
                 hist_deg,
                 n_modes: int = 6, 
                 space_dims: int = 2,
                 time_independent: bool = True,
                 key_dim: int = 16,
                 dropout = 0.,
                 n_stacked_encmodule: int = 2,
                ):
        super().__init__()        
        self.homogenizing = homogenizing
        self.n_stacked_encmodule = n_stacked_encmodule
        self.hist_deg = hist_deg
        self.n_pred_const = n_pred_const
        self.space_dims = space_dims
        self.n_modes = n_modes
        self.d_hidden = d_hidden
        
        # Encoder       
        self.a_type_embed = tf.keras.layers.Embedding(10, d_hidden, input_length = 1)
        self.a_time_embed = model_utils.MLP([d_hidden, d_hidden])
        
        if self.homogenizing:
            self.m_type_embed = tf.keras.layers.Embedding(4, d_hidden, input_length = 1)
        else:
            self.m_type_embed = tf.keras.layers.Embedding(4, d_hidden, input_length = 1)
            self.m_bound_embed = tf.keras.layers.Embedding(17, d_hidden, input_length = 2)
            self.m_inter_embed = tf.keras.layers.Embedding(3, d_hidden, input_length = 1)
        
        self.agent_embed_layer = model_utils.EmbeddingBlock([d_hidden, d_hidden])
        self.agent_rel_embed_layer = model_utils.MLP([d_hidden, d_hidden])
        self.mapel_embed_layer = model_utils.EmbeddingBlock([d_hidden, d_hidden])
        self.mapel_rel_embed_layer = model_utils.MLP([d_hidden, d_hidden])
        
        self.map_encoder = model_utils.AttnBlock(d=d_hidden, key_dim=key_dim, num_heads=n_heads, num_layers=n_layers, dropout=dropout)
        self.agent_map_encoder = [model_utils.AttnBlock(d=d_hidden, key_dim=key_dim, num_heads=n_heads, num_layers=n_layers, dropout=dropout) for _ in range(n_stacked_encmodule)]
        self.agent_social_encoder = [model_utils.AttnBlock(d=d_hidden, key_dim=key_dim, num_heads=n_heads, num_layers=n_layers, dropout=dropout) for _ in range(n_stacked_encmodule)]
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5) 
        
        # Decoder
        self.multimodal_proj = tf.keras.layers.Dense(self.n_modes*d_hidden) # model_utils.MLP([self.n_modes*d_hidden]) # 
        
        self.decoder_mean = model_utils.MLP([d_hidden, d_hidden, n_pred_const*self.space_dims])
        #self.decoder_others = model_utils.MLP([d_hidden, d_hidden, n_pred_const*2])

        self.pi = model_utils.MLP([d_hidden, d_hidden, 1])
        #self.bj = tfp.bijectors.FillScaleTriL(diag_bijector=tfp.bijectors.Exp(), diag_shift=None)

        
    @tf.function
    def _efficient_add(self, tensor1, tensor2):
        expand1 = tf.expand_dims(tensor1, axis=1)
        expand2 = tf.expand_dims(tensor2, axis=2)
        return expand1 + expand2
    
    @tf.function
    def _selected_mean(self, x, mask):
        """ calculate mean along axis=2 for selected elements
        Input:
            x [B,K,P,d]: tensor to calculate selected mean
            mask [B,K,P]: mask defining which elements to consider for the mean
        Output [B,K,d]
        """
        expanded_mask = tf.expand_dims(mask, axis=-1)
        masked_x = tf.where(expanded_mask, x, 0)
        summed_values = tf.reduce_sum(masked_x, axis=2)
        true_counts = tf.reduce_sum(tf.cast(mask, tf.float32), axis=2, keepdims=True)
        true_counts += 1e-7
        return summed_values / true_counts

class EP_F(EP_Base):
    def __init__(self, 
                 d_hidden, 
                 n_pred_const,
                 n_heads, 
                 n_layers, 
                 homogenizing,
                 hist_deg,
                 n_modes: int = 6, 
                 space_dims: int = 2,
                 time_independent: bool = True,
                 key_dim: int = 16,
                 dropout = 0.,
                 n_stacked_encmodule: int = 2,
                ):
        super().__init__(d_hidden = d_hidden, 
                         n_pred_const = n_pred_const,
                         n_heads = n_heads,
                         n_layers = n_layers,
                         homogenizing = homogenizing,
                         hist_deg = hist_deg,
                         n_modes = n_modes, 
                         space_dims = space_dims,
                         time_independent = time_independent,
                         key_dim = key_dim,
                         dropout = dropout,
                         n_stacked_encmodule = n_stacked_encmodule,
                         )
        
        self.decoder_others = model_utils.MLP([d_hidden, d_hidden, n_pred_const*2])

    @tf.function
    def call(self, data, training):
        input_agents = data["input_agents"] # [B,A,d_agent]
        input_map = data["input_map"]   #[B,M,d_map]

        mask_neighbor_agent = data["mask_neighbor_agent"]   # [B,A,A]
        mask_neighbor_mapel = data["mask_neighbor_mapel"]   # [B,M,M]
        mask_neighbor_agent_mapel = data["mask_neighbor_agent_mapel"]   # [B,A,M]
        
        agents_origin = data["agents_origin"]#[:,:,None,:]   # [B, A, 2]
        agents_theta = data["agents_theta"]#[:,:,None]       # [B, A]
        mapels_origin = data["mapels_origin"]#[:,:,None,:]   # [B, M, 2]
        mapels_theta = data["mapels_theta"]#[:,:,None]       # [B, M]
        
        B, A, M, K = input_agents.shape[0], input_agents.shape[1], input_map.shape[1], self.n_modes
        
        c_agent_theta, s_agent_theta = tf.cos(agents_theta[:, :, None]), tf.sin(agents_theta[:, :, None])
        agent_R = tf.reshape(tf.concat([c_agent_theta, -s_agent_theta, s_agent_theta, c_agent_theta], axis=-1), [B, A, 2,2])  # [B, A, 2, 2]
        agent_t = agents_origin[:, :, None, :] # [B,A,1,2]
        
        input_a_cps = tf.reshape(input_agents[:, :, :(self.hist_deg+1)*self.space_dims], (B,A,-1,2)) #[B,A,6,2]
        transformed_input_a_cps = (input_a_cps - agent_t) # @ agent_R #[B, A, 6, 2]
        
        input_a_cps_vec = input_a_cps[:, :, 1:, :] - input_a_cps[:, :, :-1, :] #[B,A,5,2]
        input_a_cps_vec = tf.reshape(input_a_cps_vec, (B,A,-1))
        input_a_time = input_agents[:, :, -3:-1]
        input_a_type = tf.cast(input_agents[:, :, -1][..., None], tf.int32)
        input_a_rel = tf.concat([agents_origin,  tf.sin(agents_theta[:, :, None]), tf.cos(agents_theta[:, :, None])], axis=2)
        
        
        if self.homogenizing:
            input_m_cps = tf.reshape(input_map[:, :, :-1], (B,M,-1,2)) #[B,M,4,2]
            input_m_type = tf.cast(input_map[:, :, -1][..., None], tf.int32)
        else:
            input_m_cps = tf.reshape(input_map[:, :, :-4], (B,M,-1,2)) #[B,M,12,2]
            input_m_bound = tf.cast(input_map[:, :, -4:-2], tf.int32)
            input_m_type = tf.cast(input_map[:, :, -2][..., None], tf.int32)
            input_m_inter = tf.cast(input_map[:, :, -1][..., None], tf.int32)
        
        input_m_cps_normlized = input_m_cps - mapels_origin[:, :, None, :]
        input_m_cps_normlized = tf.reshape(input_m_cps_normlized,(B,M,-1))
        
        input_m_rel = tf.concat([mapels_origin,  tf.sin(mapels_theta[:, :, None]), tf.cos(mapels_theta[:, :, None])], axis=2)
        
        # Forward
        a_time_embed = self.a_time_embed(input_a_time)
        a_type_embed = tf.reduce_sum(self.a_type_embed(input_a_type), axis = -2)
        a_rel_embed = self.agent_rel_embed_layer(input_a_rel)
        agents_emb = self.agent_embed_layer(input_a_cps_vec, a_type_embed+a_time_embed) # [B, A, d_hidden]
        
        
        if self.homogenizing:
            m_type_embed = tf.reduce_sum(self.m_type_embed(input_m_type), axis= -2)
            m_other_features = m_type_embed
        else:
            m_type_embed = tf.reduce_sum(self.m_type_embed(input_m_type), axis= -2)
            m_bound_embed = tf.reduce_sum(self.m_bound_embed(input_m_bound), axis= -2)
            m_inter_embed = tf.reduce_sum(self.m_inter_embed(input_m_inter), axis= -2)
            m_other_features = m_type_embed+ m_bound_embed + m_inter_embed

        m_rel_embed = self.mapel_rel_embed_layer(input_m_rel)
        mapels_emb = self.mapel_embed_layer(input_m_cps_normlized, m_other_features)   # [B, M, d_hidden]    
        
        mapels_emb = self.map_encoder(q=mapels_emb+m_rel_embed, masks=mask_neighbor_mapel, training=training) #+ mapels_time # [B, M, d_hidden]
        
        for i in range(self.n_stacked_encmodule):    
            agents_emb = self.agent_map_encoder[i](q=agents_emb+a_rel_embed, k=mapels_emb, masks=mask_neighbor_agent_mapel, training=training) # [B, A, d_hidden]
            agents_emb = self.agent_social_encoder[i](q=agents_emb, masks=mask_neighbor_agent, training=training)    # [B, A, d_hidden]
        
        agents_emb = self.layernorm1(agents_emb)
        y_const_others = self.decoder_others(agents_emb) # [B,A,d_output]
        y_const_others = tf.reshape(y_const_others, (B,A,-1, 2)) # [B,A,C,2]
        y_const_others = tf.reshape(y_const_others@agent_R, (B,A,-1)) # rotate predictions to right [B,A,2C]
        
        agents_emb = tf.reshape(self.multimodal_proj(agents_emb), (B, A, self.n_modes, -1)) # [B, A, K, d_hidden]
        consts_mean = self.decoder_mean(agents_emb)      # [B, A, K, 2*n_consts]
 
        logits = tf.squeeze(self.pi(agents_emb), axis = -1)
        
        return {'logits': logits, # [B,A,K]
                'y_const_mean': consts_mean, # [B,A,K,n_consts*2]
                'y_const_mean_others': y_const_others[:, :, None, :], # # [B,A,1,n_consts*2]
                'input_a_cps': transformed_input_a_cps, #[B, A, 6, 2]
               }


    
class EP_Q(EP_Base):
    def __init__(self, 
                 d_hidden, 
                 n_pred_const,
                 n_heads, 
                 n_layers, 
                 homogenizing,
                 hist_deg,
                 n_modes: int = 6, 
                 space_dims: int = 2,
                 key_dim: int = 16,
                 dropout = 0.,
                 n_stacked_encmodule: int = 2,
                 ):
        super().__init__(d_hidden = d_hidden, 
                         n_pred_const = n_pred_const,
                         n_heads = n_heads,
                         n_layers = n_layers,
                         homogenizing = homogenizing,
                         hist_deg = hist_deg,
                         n_modes = n_modes, 
                         space_dims = space_dims,
                         key_dim = key_dim,
                         dropout = dropout,
                         n_stacked_encmodule = n_stacked_encmodule,
                         )

    @tf.function
    def call(self, data, training):
        input_agents = data["input_agents"] # [B,A,d_agent]
        input_map = data["input_map"]   #[B,M,d_map]

        mask_neighbor_agent = data["mask_neighbor_agent"]   # [B,A,A]
        mask_neighbor_mapel = data["mask_neighbor_mapel"]   # [B,M,M]
        mask_neighbor_agent_mapel = data["mask_neighbor_agent_mapel"]   # [B,A,M]
        
        agents_origin = data["agents_origin"]#[:,:,None,:]   # [B, A, 2]
        agents_theta = data["agents_theta"]#[:,:,None]       # [B, A]
        mapels_origin = data["mapels_origin"]#[:,:,None,:]   # [B, M, 2]
        mapels_theta = data["mapels_theta"]#[:,:,None]       # [B, M]
        
        B, A, M, K = input_agents.shape[0], input_agents.shape[1], input_map.shape[1], self.n_modes
        
        c_agent_theta, s_agent_theta = tf.cos(agents_theta[:, :, None]), tf.sin(agents_theta[:, :, None])
        agent_R = tf.reshape(tf.concat([c_agent_theta, -s_agent_theta, s_agent_theta, c_agent_theta], axis=-1), [B, A, 2,2])  # [B, A, 2, 2]
        agent_R = agent_R[:, :, None, :, :]  # [B, A, 1, 2, 2]
        agent_t = agents_origin[:, :, None, None, :] # [B,A,1,1,2]
        
        input_a_cps = tf.reshape(input_agents[:, :, :(self.hist_deg+1)*self.space_dims], [B,A,-1,2]) #[B,A,6,2]
        input_a_cps = tf.repeat(input_a_cps[:, None, :, :, :], A, axis=1) #[B, A, A, 6, 2]
        input_a_cps = (input_a_cps - agent_t) @ agent_R # [B, A, A, 6, 2]

        
        input_a_cps_vec = input_a_cps[:, :, :, 1:, :] - input_a_cps[:, :, :, :-1, :] #[B,A,A,5,2]
        input_a_cps_vec = tf.reshape(input_a_cps_vec, [B,A,A,-1])
        input_a_time = input_agents[:, :, -3:-1]
        input_a_type = tf.cast(input_agents[:, :, -1][..., None], tf.int32)
        
        rel_a_a_pos = tf.squeeze(self._efficient_add(agents_origin, -agents_origin)[:, :, :, None, :] @ agent_R, axis=-2)# [B, A, A, 2]
        rel_a_a_theta = self._efficient_add(agents_theta, -agents_theta) # [B, A, A]
        input_a_rel = tf.concat([rel_a_a_pos,  tf.sin(rel_a_a_theta[..., None]), tf.cos(rel_a_a_theta[..., None])], axis=-1) # [B,A,A,4]
        
        if self.homogenizing:
            input_m_cps = tf.reshape(input_map[:, :, :-1], (B,M,-1,2)) #[B,M,4,2]
            input_m_type = tf.cast(input_map[:, :, -1][..., None], tf.int32)
        else:
            input_m_cps = tf.reshape(input_map[:, :, :-4], (B,M,-1,2)) #[B,M,4,2]
            input_m_bound = tf.cast(input_map[:, :, -4:-2], tf.int32)
            input_m_type = tf.cast(input_map[:, :, -2][..., None], tf.int32)
            input_m_inter = tf.cast(input_map[:, :, -1][..., None], tf.int32)
        
      
        input_m_cps = tf.repeat(input_m_cps[:, None, :, :, :], A, axis= 1) #[B,A,M,4,2]
        input_m_cps = (input_m_cps - agent_t) @ agent_R # [B, A, M, 4, 2]

        rel_a_m_pos = tf.squeeze(self._efficient_add(mapels_origin, -agents_origin)[:, :, :, None, :] @ agent_R, axis=-2)# [B, A, M, 2]
        rel_a_m_theta = self._efficient_add(mapels_theta, -agents_theta) # [B, A, M]

        input_m_cps_normlized = input_m_cps - rel_a_m_pos[:, :, :, None, :]
        input_m_cps_normlized = tf.reshape(input_m_cps_normlized,(B,A ,M,-1))
    
        input_m_rel = tf.concat([rel_a_m_pos,  tf.sin(rel_a_m_theta[..., None]), tf.cos(rel_a_m_theta[..., None])], axis=-1) #[B,A,M,4]
        
        select_agent_mask = tf.repeat(tf.cast(tf.linalg.diag(tf.ones(A)), dtype=bool)[None, :, :], B, axis=0) #[B, A, A]
        
        # Forward
        a_time_embed = self.a_time_embed(input_a_time)
        a_type_embed = tf.reduce_sum(self.a_type_embed(input_a_type), axis = -2)
        a_rel_embed = self.agent_rel_embed_layer(input_a_rel) #[B,A,A,d_hidden]
        agents_emb = self.agent_embed_layer(input_a_cps_vec, a_type_embed[:, None, :, :]+a_time_embed[:, None, :, :]) # [B, A, A, d_hidden]
        
        
        if self.homogenizing:
            m_type_embed = tf.reduce_sum(self.m_type_embed(input_m_type), axis= -2)
            m_other_features = m_type_embed[:, None, :, :]
        else:
            m_type_embed = tf.reduce_sum(self.m_type_embed(input_m_type), axis= -2)
            m_bound_embed = tf.reduce_sum(self.m_bound_embed(input_m_bound), axis= -2)
            m_inter_embed = tf.reduce_sum(self.m_inter_embed(input_m_inter), axis= -2)
            m_other_features = m_type_embed[:, None, :, :]+ m_bound_embed[:, None, :, :] + m_inter_embed[:, None, :, :]

        m_rel_embed = self.mapel_rel_embed_layer(input_m_rel) #[B,A,M,d_hidden]
        mapels_emb = self.mapel_embed_layer(input_m_cps_normlized, m_other_features)   # [B, A, M, d_hidden]    
        
        agents_emb = tf.reshape(agents_emb, [B*A, A, -1])
        mapels_emb = tf.reshape(mapels_emb, [B*A, M, -1])
        a_rel_embed = tf.reshape(a_rel_embed, [B*A, A, -1])
        m_rel_embed = tf.reshape(m_rel_embed, [B*A, M, -1])
        select_agent_mask = tf.reshape(select_agent_mask, [B*A*A])
        mask_neighbor_mapel = tf.reshape(tf.repeat(mask_neighbor_mapel[:, None, :, :], A, axis=1), [B*A, M, M])
        mask_neighbor_agent_mapel = tf.reshape(tf.repeat(mask_neighbor_agent_mapel[:, None, :, :], A, axis=1), [B*A, A, M])
        mask_neighbor_agent = tf.reshape(tf.repeat(mask_neighbor_agent[:, None, :, :], A, axis=1), [B*A, A, A])
        
        mapels_emb = self.map_encoder(q=mapels_emb+m_rel_embed, masks=mask_neighbor_mapel, training=training) #+ mapels_time # [B*A, M, d_hidden]
        
        for i in range(self.n_stacked_encmodule):
            # TODO: it might be problematic using static neighbor among agents or between agents and mapels, since the agents are dynamic    
            agents_emb = self.agent_map_encoder[i](q=agents_emb+a_rel_embed, k=mapels_emb, masks=mask_neighbor_agent_mapel, training=training) # [B*A, A, d_hidden]
            agents_emb = self.agent_social_encoder[i](q=agents_emb, masks=mask_neighbor_agent, training=training)    # [B*A, A, d_hidden]
       
        agents_emb = self.layernorm1(agents_emb)
        agents_emb = tf.reshape(agents_emb, [B*A*A, self.d_hidden])
        input_a_cps = tf.reshape(input_a_cps, [B*A*A, -1, 2])
        target_index = tf.squeeze(tf.where(select_agent_mask), axis= -1)

        agents_emb = tf.gather(agents_emb, target_index, axis = 0, batch_dims=0)
        input_a_cps = tf.gather(input_a_cps, target_index, axis = 0, batch_dims=0)
        
        agents_emb = tf.reshape(agents_emb, [B,A,self.d_hidden])
        input_a_cps = tf.reshape(input_a_cps, [B, A, -1, 2])
        
        
        agents_emb = tf.reshape(self.multimodal_proj(agents_emb), [B, A, self.n_modes, self.d_hidden]) # [B, A, K, d_hidden]
        
        consts_mean = self.decoder_mean(agents_emb)      # [B, A, K, 2*n_consts]
        logits = tf.squeeze(self.pi(agents_emb), axis = -1)
        
        return {'logits': logits, # [B,A,K]
                'y_const_mean': consts_mean, # [B,A,K,n_consts*2]
                'input_a_cps': input_a_cps, #[B, A, 6, 2]
               }

    
models_dict = {'EP_F': EP_F, 'EP_Q': EP_Q}


