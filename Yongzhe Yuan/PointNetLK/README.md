# 关于ModelNet40的数据集读取


### Notes: 
* 所有模型文件的格式均为.off


### Main files for experiments:
* ModelNet40.py: 数据集批量读取，内已有详细注释。
* read_off.py: 读取.off文件，并含有Mesh类用于生成mesh对象，存储每个点云的顶点信息。
* transforms.py: 内有一些点云变换代码

### How to use them:

* read_off.py和transforms.py不需要做任何改动
* 读取数据集入口查看ModelNet40.py的构造函数，需要传入三个参数：rootdir表示数据集所在根目录，pattern表示要读取数据集的模式（test or train），transform表示是否对点云进行变换
* 读取数据集出口返回一个sample对象和所有实例标签（可删）。注意！！！代码中调用时transform全部设置为True，sample返回的是利用顶点坐标信息变换后的点云（类型为Tensor），即返回的是一个经过变换后的实例
  

```
