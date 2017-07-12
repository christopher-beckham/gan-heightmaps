#!/bin/bash

env
PYTHONUNBUFFERED=1 \
THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32,nvcc.fastmath=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once \
  python experiments.py test1_nobn_bilin_both train
