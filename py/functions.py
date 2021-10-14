#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import logging
import numpy as np
import pandas as pd
import tensorflow as tf


### utils

def init_logger(file_path, file_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter('%(message)s'))
    handler2 = logging.FileHandler(filename=f'{file_path}{file_name}')
    handler2.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


@tf.autograph.experimental.do_not_convert
def read_yaml():
    yaml_path = '/tf/Kaggle-OpenVaccine-COVID-19_mRNA_Vaccine_Degradation_Prediction/yaml/config.yml'
    with open(yaml_path, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    return config


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


### data augmentation

def aug_data(df):
    config = read_yaml()
    aug_df = pd.read_csv(f'{config["base_path"]}/{config["aug_data"]}/aug_data.csv')
    target_df = df.copy()
    new_df = aug_df[aug_df['id'].isin(target_df['id'])]
                         
    del target_df['structure']
    del target_df['predicted_loop_type']
    new_df = new_df.merge(target_df, on=['id','sequence'], how='left')

    df['cnt'] = df['id'].map(new_df[['id','cnt']].set_index('id').to_dict()['cnt'])
    df['log_gamma'] = 100
    df['score'] = 1.0
    df = df.append(new_df[df.columns])
    return df


### metrics

def mcrmse(t, p):
    config = read_yaml()
    t = t[:, : config['pub_seq_len_target']]
    p = p[:, : config['pub_seq_len_target']]
    
    score = np.mean(np.sqrt(np.mean(np.mean((p - t) ** 2, axis=1), axis=0)))
    return score


def mcrmse_loss(t, y):
    config = read_yaml()
    t = t[:, : config['pub_seq_len_target']]
    y = y[:, : config['pub_seq_len_target']]
    
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.reduce_mean((t - y) ** 2, axis=1), axis=0)))
    return loss


### feature engineering

def read_bpps_max(df):
    config = read_yaml()
    bpps_arr = [
        np.load(f'{config["base_path"]}/{config["comp_data"]}/bpps/{mol_id}.npy').max(axis=1) for mol_id in df.id.to_list() 
    ]   
    return bpps_arr

def read_bpps_sum(df):
    config = read_yaml()
    bpps_arr = [
        np.load(f'{config["base_path"]}/{config["comp_data"]}/bpps/{mol_id}.npy').sum(axis=1) for mol_id in df.id.to_list()
    ]
    return bpps_arr

def read_bpps_nb(df):
    config = read_yaml()
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f'{config["base_path"]}/{config["comp_data"]}/bpps/{mol_id}.npy')
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_arr.append(bpps_nb)
    return bpps_arr


### for GNN

def get_structure_adj(train):
    Ss = []
    for i in range(len(train)):
        seq_length = train['seq_length'].iloc[i]
        structure = train['structure'].iloc[i]
        sequence = train['sequence'].iloc[i]

        cue = []
        a_structures = {
            ('A', 'U'): np.zeros([seq_length, seq_length]),
            ('C', 'G'): np.zeros([seq_length, seq_length]),
            ('U', 'G'): np.zeros([seq_length, seq_length]),
            ('U', 'A'): np.zeros([seq_length, seq_length]),
            ('G', 'C'): np.zeros([seq_length, seq_length]),
            ('G', 'U'): np.zeros([seq_length, seq_length]),
        }
        a_structure = np.zeros([seq_length, seq_length])
        for i in range(seq_length):
            if structure[i] == '(':
                cue.append(i)
            elif structure[i] == ')':
                start = cue.pop()
                a_structures[(sequence[start], sequence[i])][start, i] = 1
                a_structures[(sequence[i], sequence[start])][i, start] = 1
        
        a_strc = np.stack([a for a in a_structures.values()], axis=2)
        a_strc = np.sum(a_strc, axis=2, keepdims=True)
        Ss.append(a_strc)
    
    Ss = np.array(Ss)
    return Ss


def get_distance_matrix(As):
    idx = np.arange(As.shape[1])
    Ds = [np.abs(idx[i] - idx) for i in range(len(idx))]
    Ds = 1 / (np.array(Ds) + 1)
    Ds = Ds[None, :, :]
    Ds = np.repeat(Ds, len(As), axis=0)
    Dss = [Ds ** i for i in [1, 2, 4]]
    Ds = np.stack(Dss, axis=3)
    return Ds


def return_ohe(n, i):
    tmp = [0] * n
    tmp[i] = 1
    return tmp


def get_input(train):
    mapping = {}
    vocab = ['A', 'G', 'C', 'U']
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_node = np.stack(train['sequence'].apply(lambda x: list(map(lambda y: mapping[y], list(x)))))

    mapping = {}
    vocab = ['S', 'M', 'I', 'B', 'H', 'E', 'X']
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_loop = np.stack(train['predicted_loop_type'].apply(lambda x: list(map(lambda y: mapping[y], list(x)))))
    
    mapping = {}
    vocab = ['.', '(', ')']
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_node = np.concatenate([X_node, X_loop], axis=2)
    
    a = np.sum(X_node * (2 ** np.arange(X_node.shape[2])[None, None, :]), axis=2)
    vocab = sorted(set(a.flatten()))
    ohes = [a == v for v in vocab]
    ohes = np.stack(ohes, axis=2)
    
    bpps_sum_fea = np.array(train['bpps_sum'].to_list())[:, :, np.newaxis]
    bpps_max_fea = np.array(train['bpps_max'].to_list())[:, :, np.newaxis]
    bpps_nb_fea = np.array(train['bpps_nb'].to_list())[:, :, np.newaxis]
    
    X_node = np.concatenate([X_node, ohes, bpps_sum_fea, bpps_max_fea, bpps_nb_fea], axis=2).astype(np.float32)
    return X_node


### create dataset

def read_data():
    config = read_yaml()

    train = pd.read_json(f'{config["base_path"]}/{config["comp_data"]}/train.json', lines=True)
    train['bpps_sum'] = read_bpps_sum(train)
    train['bpps_max'] = read_bpps_max(train)
    train['bpps_nb'] = read_bpps_nb(train)
    train = train[train.signal_to_noise > 1].reset_index(drop=True)
    train = aug_data(train)

    test  = pd.read_json(f'{config["base_path"]}/{config["comp_data"]}/test.json', lines=True)
    test['bpps_sum'] = read_bpps_sum(test)
    test['bpps_max'] = read_bpps_max(test)
    test['bpps_nb'] = read_bpps_nb(test)
    test = aug_data(test)
    test_pub = test[test['seq_length'] == config['pub_seq_len']]
    test_pri = test[test['seq_length'] == config['pri_seq_len']]

    As = np.array([np.load(f'{config["base_path"]}/{config["comp_data"]}/bpps/{id}.npy') for id in train['id']])
    As_pub = np.array([np.load(f'{config["base_path"]}/{config["comp_data"]}/bpps/{id}.npy') for id in test_pub['id']])
    As_pri = np.array([np.load(f'{config["base_path"]}/{config["comp_data"]}/bpps/{id}.npy') for id in test_pri['id']])

    sub = pd.read_csv(f'{config["base_path"]}/{config["comp_data"]}/sample_submission.csv')

    return train, test_pub, test_pri, As, As_pub, As_pri, sub


def create_dataset(pred=False):
    config = read_yaml()

    train, test_pub, test_pri, As, As_pub, As_pri, sub = read_data()
    targets = list(sub.columns[1:])
    ignore = -10000
    ignore_length = config['pub_seq_len'] - config['pub_seq_len_target']
    y_train = []
    for target in targets:
        y = np.vstack(train[target])
        dummy = np.zeros([y.shape[0], ignore_length]) + ignore
        y = np.hstack([y, dummy])
        y_train.append(y)
    y = np.stack(y_train, axis=2)

    Ss = get_structure_adj(train)
    Ss_pub = get_structure_adj(test_pub)
    Ss_pri = get_structure_adj(test_pri)
    Ds = get_distance_matrix(As)
    Ds_pub = get_distance_matrix(As_pub)
    Ds_pri = get_distance_matrix(As_pri)
    
    As = np.concatenate([As[:,:,:,None], Ss, Ds], axis=3).astype(np.float32)
    As_pub = np.concatenate([As_pub[:,:,:,None], Ss_pub, Ds_pub], axis=3).astype(np.float32)
    As_pri = np.concatenate([As_pri[:,:,:,None], Ss_pri, Ds_pri], axis=3).astype(np.float32)
    del Ss, Ds, Ss_pub, Ds_pub, Ss_pri, Ds_pri

    X_node = get_input(train)
    X_node_pub = get_input(test_pub)
    X_node_pri = get_input(test_pri)
    
    if pred:
        return X_node, X_node_pub, X_node_pri, As, As_pub, As_pri, targets, test_pub, test_pri
    else:
        return X_node, X_node_pub, X_node_pri, As, As_pub, As_pri, y