## MAE TEST


### TIPS 

启动 tensorboard: 

```python 
# 此时会将信息输出到 logs 文件夹里面
# --port 可以不加，不加的话默认 6006 
tensorboard --logdir=logs --port=8888
# 默认是 6006 端口进行显示
http://localhost:6006/
```



### 配环境 

```python
# 配的 timm 版本是这个，然后有些 qk_scale=None 的地方需要删掉 
pip install timm==0.4.12

# tensorboard 中下面这个会报错，直接删掉 
LooseVersion = distutils.version.LooseVersion
```


### 运行指令 & 说明 


#### MAE baseline 说明 

finetune: 
```python
# 运行指令：
python main_finetune.py --finetune 模型路径 
```


#### EMA-MAE 说明 
pretrain:
```python 
tensorboard --logdir=bootstrap_logs/
python ema_bootstrap_pretrain.py --warmup_target_epochs=10 
```

finetune: 
```python 
# Example: 
python main_finetune.py --finetune=bootstrap_pretrain_weights/80_0.005/checkpoint-199.pth --output_dir=bootstrap_finetuned_weights --log_dir=bootstrap_finetune_logs --additional_info=80_0.005 --device=cuda:0
```

#### K-MAE 说明 
pretrain: 
```python
tensorboard --logdir=k_logs
python k_bootstrap_pretrain.py --num_k=1 
```






### sketch board
```python 
python main_finetune.py --finetune=bootstrap_pretrain_weights/10_0.005/checkpoint-199.pth --output_dir=bootstrap_finetuned_weights/10_0.005
```