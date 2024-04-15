#!/bin/bash


python k_bootstrap_pretrain.py --device=cuda:2 --num_k=10
python k_bootstrap_pretrain.py --device=cuda:2 --num_k=20
python k_bootstrap_pretrain.py --device=cuda:2 --num_k=40
python k_bootstrap_pretrain.py --device=cuda:2 --num_k=50