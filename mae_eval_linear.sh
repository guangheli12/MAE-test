#!/bin/bash

python main_linprobe.py --finetune=baseline_pretrain_weights/checkpoint-199.pth --output_dir=baseline_linear_weights --log_dir=baseline_linear_logs --device=cuda:0