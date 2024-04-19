#!/bin/bash



# python new_pretrain.py --num_k=10 --output_dir=new_pretrain_weights --log_dir=new_pretrain_logs
# python new_pretrain.py --num_k=20 --output_dir=new_pretrain_weights --log_dir=new_pretrain_logs
python main_finetune.py --finetune=new_pretrain_weights/10_0.005/checkpoint-199.pth --output_dir=new_finetuned_weights --log_dir=new_finetune_logs --additional_info=10_0.005 --device=cuda:0
python main_finetune.py --finetune=new_pretrain_weights/20_0.005/checkpoint-199.pth --output_dir=new_finetuned_weights --log_dir=new_finetune_logs --additional_info=20_0.005 --device=cuda:0
# python main_finetune.py --finetune=bootstrap_pretrain_weights/5_0.005/checkpoint-199.pth --output_dir=bootstrap_finetuned_weights --log_dir=bootstrap_finetune_logs --additional_info=5_0.005 --device=cuda:0