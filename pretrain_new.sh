#!/bin/bash

# python new_pretrain.py --num_k=10 --output_dir=new_pretrain_weights --log_dir=new_pretrain_logs
# python new_pretrain.py --num_k=20 --output_dir=new_pretrain_weights --log_dir=new_pretrain_logs
# python new_pretrain.py --num_k=5 --output_dir=new_pretrain_weights --log_dir=new_pretrain_logs

python k_bootstrap_pretrain.py --num_k=5 --output_dir=k_pretrain_weights --log_dir=k_pretrain_logs