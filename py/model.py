#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from functions import mcrmse_loss


def attention(x_inner, x_outer, n_factor, dropout):
    x_Q =  L.Conv1D(n_factor, 1, activation='linear', 
                    kernel_initializer='glorot_uniform',
                    bias_initializer='glorot_uniform'
                    )(x_inner)
    x_K =  L.Conv1D(n_factor, 1, activation='linear', 
                    kernel_initializer='glorot_uniform',
                    bias_initializer='glorot_uniform'
                    )(x_outer)
    x_V =  L.Conv1D(n_factor, 1, activation='linear', 
                    kernel_initializer='glorot_uniform',
                    bias_initializer='glorot_uniform'
                    )(x_outer)
    x_KT = L.Permute((2, 1))(x_K)
    res = L.Lambda(lambda c: K.batch_dot(c[0], c[1]) / np.sqrt(n_factor))([x_Q, x_KT])
    att = L.Lambda(lambda c: K.softmax(c, axis=-1))(res)
    att = L.Lambda(lambda c: K.batch_dot(c[0], c[1]))([att, x_V])
    return att


def multi_head_attention(x, y, n_factor, n_head, dropout):
    if n_head == 1:
        att = attention(x, y, n_factor, dropout)
    else:
        n_factor_head = n_factor // n_head
        heads = [attention(x, y, n_factor_head, dropout) for i in range(n_head)]
        att = L.Concatenate()(heads)
        att = L.Dense(n_factor, 
                      kernel_initializer='glorot_uniform',
                      bias_initializer='glorot_uniform',
                     )(att)
    x = L.Add()([x, att])
    x = L.LayerNormalization()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    return x


def res(x, unit, kernel=3, rate=0.1):
    h = L.Conv1D(unit, kernel, 1, padding='same', activation=None)(x)
    h = L.LayerNormalization()(h)
    h = L.LeakyReLU()(h)
    h = L.Dropout(rate)(h)
    return L.Add()([x, h])


def forward(x, unit, kernel=3, rate=0.1):
    h = L.Conv1D(unit, kernel, 1, padding='same', activation=None)(x)
    h = L.LayerNormalization()(h)
    h = L.Dropout(rate)(h)
    h = L.LeakyReLU()(h)
    h = res(h, unit, kernel, rate)
    return h


def adj_attn(x, adj, unit, n=2, rate=0.1):
    x_a = x
    x_as = []
    for i in range(n):
        x_a = forward(x_a, unit)
        x_a = tf.matmul(adj, x_a)
        x_as.append(x_a)
    if n == 1:
        x_a = x_as[0]
    else:
        x_a = L.Concatenate()(x_as)
    x_a = forward(x_a, unit)
    return x_a


def get_base(node_dim, adj_dim):
    node = tf.keras.Input(shape=(None, node_dim), name='node')
    adj = tf.keras.Input(shape=(None, None, adj_dim), name='adj')
    
    adj_learned = L.Dense(1, 'relu')(adj)
    adj_all = L.Concatenate(axis=3)([adj, adj_learned])
        
    xs = []
    xs.append(node)
    x1 = forward(node, 128, kernel=3, rate=0.0)
    x2 = forward(x1, 64, kernel=6, rate=0.0)
    x3 = forward(x2, 32, kernel=15, rate=0.0)
    x4 = forward(x3, 16, kernel=30, rate=0.0)
    x = L.Concatenate()([x1, x2, x3, x4])
    
    for unit in [64, 32]:
        x_as = []
        for i in range(adj_all.shape[3]):
            x_a = adj_attn(x, adj_all[:, :, :, i], unit, rate=0.0)
            x_as.append(x_a)
        x_c = forward(x, unit, kernel=30)
        
        x = L.Concatenate()(x_as + [x_c])
        x = forward(x, unit)
        x = multi_head_attention(x, x, unit, 4, 0.0)
        xs.append(x)
        
    x = L.Concatenate()(xs)

    model = tf.keras.Model(inputs=[node, adj], outputs=[x])
    return model


def get_ae_model(base, node_dim, adj_dim):
    node = tf.keras.Input(shape=(None, node_dim), name='node')
    adj = tf.keras.Input(shape=(None, None, adj_dim), name='adj')

    x = base([L.SpatialDropout1D(0.3)(node), adj])
    x = forward(x, 64, rate=0.3)
    p = L.Dense(node_dim, 'sigmoid')(x)
    
    loss = - tf.reduce_mean(20 * node * tf.math.log(p + 1e-4) + (1 - node) * tf.math.log(1 - p + 1e-4))
    model = tf.keras.Model(inputs=[node, adj], outputs=[loss])
    
    model.compile(optimizer='adam', loss=lambda t, y: y)
    return model


def get_model(base, node_dim, adj_dim, model_type=0):
    node = tf.keras.Input(shape=(None, node_dim), name='node')
    adj = tf.keras.Input(shape=(None, None, adj_dim), name='adj')
    
    x = base([node, adj])
    x = forward(x, 128, rate=0.4)
    if model_type == 0:
        x = L.Bidirectional(L.LSTM(128, return_sequences=True, dropout=0.1))(x)
        x = L.Bidirectional(L.LSTM(128, return_sequences=True, dropout=0.1))(x)
    elif model_type == 1:
        x = L.Bidirectional(L.GRU(128, return_sequences=True, dropout=0.1))(x)
        x = L.Bidirectional(L.GRU(128, return_sequences=True, dropout=0.1))(x)
    elif model_type == 2:
        x = L.Bidirectional(L.LSTM(128, return_sequences=True, dropout=0.1))(x)
        x = L.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    elif model_type == 3:
        x = L.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = L.Dense(64, activation='relu')(x)
    x = L.Dense(5, None)(x)

    model = tf.keras.Model(inputs=[node, adj], outputs=[x])
    
    model.compile(optimizer='adam', loss=mcrmse_loss)
    return model