#!/bin/bash

# tau = 0.1, 0.3 
python ema_bootstrap_pretrain.py --tau=0.1 --warmup_target_epochs=5 --output_dir=bootstrap_pretrain_weights --log_dir=bootstrap_pretrain_logs --device=cuda:1
python ema_bootstrap_pretrain.py --tau=0.3 --warmup_target_epochs=5 --output_dir=bootstrap_pretrain_weights --log_dir=bootstrap_pretrain_logs --device=cuda:1