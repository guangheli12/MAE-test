#!/bin/bash
python main_finetune.py --finetune=bootstrap_pretrain_weights/5_0.005/checkpoint-199.pth --log_dir=bootstrap_finetune_logs --output_dir=bootstrap_finetune_weights --additional_info=5_0.005 --device=cuda:0