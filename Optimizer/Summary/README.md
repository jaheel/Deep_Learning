# 随机梯度下降法SGD

* 用小批量样本的平均损失代替全体样本的平均损失进行参数更新，可以加快参数更新频率，加速收敛。
* 小批量样本的平均损失是全体样本平均损失的无偏估计。



> 训练集中的同类样本是相关的，同类样本中不同个体的损失是相似的，所以随机抽取的一个样本损失可以作为该类所有样本损失的估计。



前提：定义一些相关变量和学习率，并随机生成训练集

算法步骤(epoch)：

1. 获得随机打乱的整数序列
2. 整体打乱训练集
3. 顺序取出batch个样本
4. 计算梯度
5. 更新参数



基于SGD的改进算法，只有第5步的参数更新不同，前面都一样的，后面只给出更新参数的程序段。

# 基本动量法

模拟小球的滚动过程来加速神经网络的收敛。

速度变量积累了历史梯度信息，使之具有惯性，当梯度方向一致时，加速收敛；当梯度方向不一致时，减小路径曲折程度。



```python
mu = 0.9
v = mu * v # v:速度，初始化为0. 该语句表明v累加了历史梯度
v += - lr * dx # 速度v受当前梯度dx调节
x += v
```



mu越大，摩擦越大；mu=1，没有摩擦；mu=0，摩擦无穷大，变为基本的梯度下降法。



前进方向：由下降方向的历史累积v和当前点的梯度方向dx合成的。



# Nesterov动量法

Nesterov Accelerated Gradient，简称NAG



```python
mu = 0.9
pre_v = v
v = mu * v
v += - lr * dx
x += v + mu * (v - pre_v)
```



# AdaGrad

前面3种方法都是对参数向量进行整体操作，对向量每个元素的调整都是一样的。



自适应算法。



```python
cache += dx ** 2
x += - (lr / (np.sqrt(cache) + eps)) * dx
```

PS：

1. 变量cache初始化为0，它和梯度向量dx同维度，每个元素累加了对应梯度元素的历史平方和。
2. cache用来归一化参数以更新步长，归一化是逐元素进行的。
3. 对于高梯度值的权重，其历史累计和大，等效学习率减小，更新强度减弱；对于低梯度值的权重，其历史累计和小，等效学习率增加，更新强度增强。
4. 除数加小常量eps可以防止除数为0。



dx为0时，更新停止，小球不可能冲出当前平地，即鞍点。



# RMSProp

采用指数衰减的方式，让cache丢弃遥远过去的历史梯度信息，只对最近的历史梯度信息进行累加，使之不那么激进，单调地降低学习率。

修改方式：使用一个梯度平方的指数加权的移动平均：

```python
decay_rate = 0.9
cache = decay_rate * cache + (1 - decay_rate) * (dx ** 2)
x += - (lr / (np.sqrt(cache) + eps)) * dx
```



# Adam

看起来像是RMSProp的动量版。



```python
mu = 0.9
decay_rate = 0.999
eps = 1e-8
v = mu * v + (1-mu) * dx
vt = v/(1 - mu ** t)
cache = decay_rate * cache + (1-decay_rate) * (dx ** 2)
cachet = cache/(1 - decay_rate ** t)
x += - (lr/ (np.sqrt(cachet) + eps)) * vt
```



# AmsGrad

Adam存在可能不收敛的缺点，因为其有效学习率为lr/(np.sqrt(cache) + eps)，其中cache主要受最近的梯度历史信息影响，故其值波动较大。当它取比较小的值时，会使有效学习率很大，使之不能收敛。

改进：使有效学习率不饿能增加，只需在Adam方法种进行很小的修改。

```python
cache = np.max((cache, decay_rate * cache + (1 - decay_rate) * (dx ** 2)))
```



# 学习率退火

随着训练次数的增加，学习率会逐渐变小。

方式：

1. 随训练周期衰减。

   > 每进行几个周期(epoch)，降低一次学习率。
   >
   > 典型做法：每5个周期学习率减少一半，或者每20个周期减少到0.1。

2. 指数衰减。

   > 数学公式是$a=a_0 e^{-kt}$，其中$a_0$和$k$是超参数，$t$是迭代次数。

3. 反比衰减。

   > 数学公式是$a=a_0/{(1+kt)}$，其中$a_0$和$k$是超参数，$t$是迭代次数。



# 参数初始化

* 小随机数初始化

  > 权重初始值接近0但不能等于0。
  >
  > 采用小的正态分布的随机数$W=0.01 \times np.random.randn(D,H)$来打破对称性。但并不是小数值一定会得到好的结果，这样会减慢收敛速度。因为在梯度反向传播的时候，会计算出非常小的梯度。
  >
  > 使用$1/sqrt(n)$来校准方差，使输出神经元的方差为1，参数初始化为$w=np.random.randn(n)/sqrt(n)$，其中n使神经元连接的输入神经元数量。

* 偏置初始化

  > 通常将偏置初始化为0或者小常数(0.01)
  >
  > 当前推荐的初始化：
  >
  > 使用ReLU时，用$w=np.random.randn(n) \times sqrt(2.0/n)$来进行权重初始化
  >
  > ```python
  > in_depth = 128
  > out_depth = 32
  > std = np.sqrt(2/in_depth)
  > weights = std * np.random.randn(in_depth, out_depth)
  > bias = np.zeros((1, out_depth))
  > ```



# 超参数调优

最重要的超参数：初始学习率、正则化系数（L2惩罚）

其他的：dropout
