#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tensorflow as tf
from functions import init_logger, read_yaml, seed_everything, mcrmse, create_dataset
from model import get_base, get_model


FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('model_type', 0, 'select model type')

config = read_yaml()
logger = init_logger(f'{config["base_path"]}/trained_model/model{FLAGS.model_type}/',
                     f'model_type{FLAGS.model_type}_train.log')

seed_everything(config['seed'])
X_node, X_node_pub, X_node_pri, As, As_pub, As_pri, y = create_dataset()
node_dim =  X_node.shape[2]
adj_dim = As.shape[3]

kfold = KFold(config['folds'], shuffle=True, random_state=config['seed'])

scores = []
preds = np.zeros([len(X_node), X_node.shape[1], 5])
for i, (tr_idx, va_idx) in enumerate(kfold.split(X_node, As)):
    logger.info(f'------ fold {i} start -----')
    X_node_tr = X_node[tr_idx]
    X_node_va = X_node[va_idx]
    As_tr = As[tr_idx]
    As_va = As[va_idx]
    y_tr = y[tr_idx]
    y_va = y[va_idx]
    
    base = get_base(node_dim, adj_dim)
    logger.info('****** load ae model ******')
    base.load_weights(f'{config["base_path"]}/pretrained_model/base_ae')
    model = get_model(base, node_dim, adj_dim, model_type=FLAGS.model_type)
    logger.info(f'model type{FLAGS.model_type}')
    for epochs, batch_size in zip(config['epochs_list'], config['batch_size_list']):
        logger.info(f'epochs : {epochs}, batch_size : {batch_size}')
        model.fit([X_node_tr, As_tr], [y_tr],
                  validation_data=([X_node_va, As_va], [y_va]),
                  epochs=epochs,
                  batch_size=batch_size, 
                  validation_freq=3)
        
    model.save_weights(f'{config["base_path"]}/trained_model/model{FLAGS.model_type}/model_type{FLAGS.model_type}_{i}')
    p = model.predict([X_node_va, As_va])
    scores.append(mcrmse(y_va, p))
    logger.info(f'fold {i}: mcrmse {scores[-1]}')
    preds[va_idx] = p
        
pd.to_pickle(preds, f'{config["base_path"]}/trained_model/model{FLAGS.model_type}/model_type{FLAGS.model_type}_oof.pkl')
logger.info(f'over all cv score: {np.mean(scores)}')