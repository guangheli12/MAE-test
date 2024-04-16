#!/bin/bash


python main_finetune.py --finetune=bootstrap_pretrain_weights/1_0.005/checkpoint-199.pth --output_dir=bootstrap_finetuned_weights --log_dir=bootstrap_finetune_logs --additional_info=1_0.005 --device=cuda:0
python main_finetune.py --finetune=bootstrap_pretrain_weights/5_0.005/checkpoint-199.pth --output_dir=bootstrap_finetuned_weights --log_dir=bootstrap_finetune_logs --additional_info=5_0.005 --device=cuda:0