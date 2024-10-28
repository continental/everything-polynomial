import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class AttnBlock(tf.keras.layers.Layer):
    ''' self-attention module
    '''
    def __init__(self, d, key_dim, num_heads, num_layers, attention_axes = None, dropout = 0.):
        '''
        Input:
            d: feature dimension
        '''
        super().__init__()
        self.attention_layers = [layers.MultiHeadAttention(num_heads, key_dim, attention_axes = attention_axes, dropout = dropout, use_bias=False) for _ in range(num_layers)]

        self.ff_layers = [tf.keras.Sequential([layers.Dense(4*d, activation='relu'),
                                               tf.keras.layers.Dropout(dropout),
                                               layers.Dense(d, activation=None)]) for _ in range(num_layers)]
        
        self.ff_prenorm_layers = [layers.LayerNormalization(epsilon=1e-5) for _ in range(num_layers)]
        
        self.attn_q_prenorm = layers.LayerNormalization(epsilon=1e-5) 
        self.attn_k_prenorm = layers.LayerNormalization(epsilon=1e-5) 

        
    @tf.function
    def call(self, q, masks, training, k = None):
        '''
        Input:
            mapels: input features as q,k,v
            masks: masks preventing attention to certain positions
        '''
        x_q = self.attn_q_prenorm(q)
        if k is None: # self attention
            for attention_layer, ff_layer, ff_prenorm_layer in zip(self.attention_layers, self.ff_layers, self.ff_prenorm_layers):
                attn_output = attention_layer(x_q, x_q, attention_mask=masks, training=training)
                x_q = x_q + attn_output
                x_q = x_q + ff_layer(ff_prenorm_layer(x_q), training = training)
        else: # cross attention
            x_k = self.attn_k_prenorm(k)
            for attention_layer, ff_layer, ff_prenorm_layer in zip(self.attention_layers, self.ff_layers, self.ff_prenorm_layers):
                attn_output = attention_layer(x_q, x_k, attention_mask=masks, training=training)
                x_q = x_q + attn_output
                x_q = x_q + ff_layer(ff_prenorm_layer(x_q), training = training)
        
        return x_q
    
    
class EmbeddingBlock(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.dense_layers = MLP(units)
        self.output_layer = MLP(units)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        
    @tf.function
    def call(self, inputs, categorical_embs):
        x = self.dense_layers(inputs)
        return self.output_layer(self.norm(x + categorical_embs))
    
    
class MLP(tf.keras.layers.Layer):
    def __init__(self, units, output_act = None):
        super().__init__()
        self.dense_layers = [tf.keras.Sequential([layers.Dense(unit, activation='relu'),
                                                  layers.LayerNormalization(epsilon=1e-5)]) for unit in units[:-1]]
        self.output_layer = layers.Dense(units[-1], activation = output_act)

    @tf.function
    def call(self, inputs):
        x = inputs
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        return self.output_layer(x)
    