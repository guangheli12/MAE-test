#!/bin/bash

# EMA-MAE(best results for finetune evaluation)
python ema_bootstrap_pretrain.py --tau=0.1 --warmup_target_epochs=5 --output_dir=bootstrap_pretrain_weights --log_dir=bootstrap_pretrain_logs --device=cuda:0

# EMA-MAE(best results for linear evaluation)
python ema_bootstrap_pretrain.py --tau=0.005 --warmup_target_epochs=1 --output_dir=bootstrap_pretrain_weights --log_dir=bootstrap_pretrain_logs --device=cuda:0