#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
from functions import read_yaml, seed_everything, create_dataset
from model import get_base, get_ae_model


config = read_yaml()

seed_everything(config['seed'])
X_node, X_node_pub, X_node_pri, As, As_pub, As_pri, _ = create_dataset()
node_dim =  X_node.shape[2]
adj_dim = As.shape[3]

base = get_base(node_dim, adj_dim)
ae_model = get_ae_model(base, node_dim, adj_dim)
for i in range(config['ae_epochs']//config['ae_epochs_each']):
    print(f'------ {i} ------')
    print('--- train ---')
    ae_model.fit([X_node, As], [X_node[:,0]],
                epochs=config['ae_epochs_each'],
                batch_size=config['ae_batch_size'])
    print('--- public ---')
    ae_model.fit([X_node_pub, As_pub], [X_node_pub[:,0]],
                epochs=config['ae_epochs_each'],
                batch_size=config['ae_batch_size'])
    print('--- private ---')
    ae_model.fit([X_node_pri, As_pri], [X_node_pri[:,0]],
                epochs=config['ae_epochs_each'],
                batch_size=config['ae_batch_size'])
    gc.collect()
print('****** save ae model ******')
base.save_weights(f'{config["base_path"]}/pretrained_model/base_ae')