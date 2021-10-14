#!/bin/bash

mkdir output
mkdir pretrained_model
mkdir -p trained_model/{model0,model1,model2,model3}


python py/pretrain.py

python py/train.py --model_type 0
python py/train.py --model_type 1
python py/train.py --model_type 2
python py/train.py --model_type 3

python py/predict.py --model_type 0
python py/predict.py --model_type 1
python py/predict.py --model_type 2
python py/predict.py --model_type 3

python py/ensemble.py

# cd output
# kaggle competitions submit -c stanford-covid-vaccine -f submission.csv -m 'late_sub'