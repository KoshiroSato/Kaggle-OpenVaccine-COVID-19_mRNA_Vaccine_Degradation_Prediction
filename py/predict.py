#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from functions import read_yaml, create_dataset
from model import get_base, get_model


FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('model_type', 0, 'select model type')

config = read_yaml()

X_node, X_node_pub, X_node_pri, As, As_pub, As_pri, targets, test_pub, test_pri = create_dataset(pred=True)
node_dim =  X_node.shape[2]
adj_dim = As.shape[3]

p_pub = 0
p_pri = 0
for i in range(config['folds']):
    K.clear_session()
    base = get_base(node_dim, adj_dim)
    model = get_model(base, node_dim, adj_dim, model_type=FLAGS.model_type)
    model.load_weights(f'{config["base_path"]}/trained_model/model{FLAGS.model_type}/model_type{FLAGS.model_type}_{i}')
    p_pub += model.predict([X_node_pub, As_pub]) / config['folds']
    p_pri += model.predict([X_node_pri, As_pri]) / config['folds']

for i, target in enumerate(targets):
    test_pub[target] = [list(p_pub[k, :, i]) for k in range(p_pub.shape[0])]
    test_pri[target] = [list(p_pri[k, :, i]) for k in range(p_pri.shape[0])]

preds_ls = []
for df, preds in [(test_pub, p_pub), (test_pri, p_pri)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=targets)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls).groupby('id_seqpos').mean().reset_index()
preds_df = preds_df[config['pred_cols']]
preds_df.to_csv(f'{config["base_path"]}/output/model_type{FLAGS.model_type}_preds.csv', index=False)