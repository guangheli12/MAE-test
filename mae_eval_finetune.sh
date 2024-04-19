#!/bin/bash

python main_finetune.py --finetune=baseline_pretrain_weights/checkpoint-199.pth --output_dir=baseline_finetune_weights --log_dir=baseline_finetune_logs --device=cuda:0