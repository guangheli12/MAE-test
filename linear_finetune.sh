#!/bin/bash

# 一个大概不到 30 分钟 


python main_linprobe.py --finetune=baseline_pretrain_weights/checkpoint-199.pth --log_dir=baseline_linear_logs --device=cuda:1

# python main_linprobe.py --finetune=bootstrap_pretrain_weights/1_0.005/checkpoint-199.pth --output_dir=bootstrap_finetuned_weights --log_dir=bootstrap_finetune_logs --additional_info=1_0.005 --device=cuda:0
# python main_linprobe.py --finetune=bootstrap_pretrain_weights/5_0.005/checkpoint-199.pth --output_dir=bootstrap_finetuned_weights --log_dir=bootstrap_finetune_logs --additional_info=5_0.005 --device=cuda:0
# python main_linprobe.py --finetune=bootstrap_pretrain_weights/10_0.005/checkpoint-199.pth --output_dir=bootstrap_finetuned_weights --log_dir=bootstrap_finetune_logs --additional_info=10_0.005 --device=cuda:0
# python main_linprobe.py --finetune=bootstrap_pretrain_weights/20_0.005/checkpoint-199.pth --output_dir=bootstrap_finetuned_weights --log_dir=bootstrap_finetune_logs --additional_info=20_0.005 --device=cuda:0
# python main_linprobe.py --finetune=bootstrap_pretrain_weights/40_0.005/checkpoint-199.pth --output_dir=bootstrap_finetuned_weights --log_dir=bootstrap_finetune_logs --additional_info=40_0.005 --device=cuda:0
# python main_linprobe.py --finetune=bootstrap_pretrain_weights/80_0.005/checkpoint-199.pth --output_dir=bootstrap_finetuned_weights --log_dir=bootstrap_finetune_logs --additional_info=80_0.005 --device=cuda:0


# python main_linprobe.py --finetune=k_pretrain_weights/1/checkpoint-199.pth --output_dir=k_linear_weights --log_dir=k_linear_logs --additional_info=1 --device=cuda:0
# python main_linprobe.py --finetune=k_pretrain_weights/2/checkpoint-199.pth --output_dir=k_linear_weights --log_dir=k_linear_logs --additional_info=2 --device=cuda:0
# python main_linprobe.py --finetune=k_pretrain_weights/4/checkpoint-199.pth --output_dir=k_linear_weights --log_dir=k_linear_logs --additional_info=4 --device=cuda:0
# python main_linprobe.py --finetune=k_pretrain_weights/10/checkpoint-199.pth --output_dir=k_linear_weights --log_dir=k_linear_logs --additional_info=10 --device=cuda:0
# python main_linprobe.py --finetune=k_pretrain_weights/20/checkpoint-199.pth --output_dir=k_linear_weights --log_dir=k_linear_logs --additional_info=20 --device=cuda:0
# python main_linprobe.py --finetune=k_pretrain_weights/40/checkpoint-199.pth --output_dir=k_linear_weights --log_dir=k_linear_logs --additional_info=40 --device=cuda:0
# python main_linprobe.py --finetune=k_pretrain_weights/50/checkpoint-199.pth --output_dir=k_linear_weights --log_dir=k_linear_logs --additional_info=50 --device=cuda:0




