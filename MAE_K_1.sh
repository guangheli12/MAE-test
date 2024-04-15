#!/bin/bash


python k_bootstrap_pretrain.py --device=cuda:1 --num_k=1
python k_bootstrap_pretrain.py --device=cuda:1 --num_k=2
python k_bootstrap_pretrain.py --device=cuda:1 --num_k=4