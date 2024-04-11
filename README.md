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