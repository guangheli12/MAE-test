## MAE TEST

Coding report is [here](/Report/report.pdf)


The repo contains 6 bash scripts of EMA-MAE(bootstrap-MAE) and baseline-MAE 


### new-MAE
To run new-MAE(An improved version of bootstrap-MAE): 

```python 
# pretrain 
python new_pretrain.py --num_k=5 --tau=0.005 --output_dir=new_pretrain_weights --log_dir=new_pretrain_logs  

# finetune evaluation 
python main_finetune.py --finetune=new_pretrain_weights/5_0.005/checkpoint-199.pth --output_dir=new_finetune_weights --log_dir=new_finetune_logs --additional_info=5_0.005 --device=cuda:0

# linear evaluation 
python main_linprobe.py --finetune=new_pretrain_weights/5_0.005/checkpoint-199.pth --output_dir=new_linear_weights --log_dir=new_linear_logs --additional_info=5_0.005 --device=cuda:0
```