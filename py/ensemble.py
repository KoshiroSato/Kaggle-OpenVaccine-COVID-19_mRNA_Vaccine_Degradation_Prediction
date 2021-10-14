#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from functions import read_yaml


config = read_yaml()

sub0 = pd.read_csv(f'{config["base_path"]}/output/model_type0_preds.csv')
sub1 = pd.read_csv(f'{config["base_path"]}/output/model_type1_preds.csv')
sub2 = pd.read_csv(f'{config["base_path"]}/output/model_type2_preds.csv')
sub3 = pd.read_csv(f'{config["base_path"]}/output/model_type3_preds.csv')

sub = sub0.copy()
sub[config['pred_cols'][1:]] = (sub0[config['pred_cols'][1:]] + 
                                sub1[config['pred_cols'][1:]] + 
                                sub2[config['pred_cols'][1:]] + 
                                sub3[config['pred_cols'][1:]]) / 4

sub.to_csv(f'{config["base_path"]}/output/submission.csv', index=False)