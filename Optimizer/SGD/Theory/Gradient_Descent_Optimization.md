# Gradient Descent Optimization

原理：目标函数$J(\theta)$关于参数$\theta$的梯度将是损失函数(loss function)上升最快的方向。

目标：最小化loss，将参数沿着梯度相反的方向前进一个步长，就可实现目标函数(loss function)的下降。步长$\alpha$称为学习率，$\nabla J(\theta)$是参数的梯度。
$$
\theta \longleftarrow \theta-\alpha \cdot \nabla J(\theta)
$$


​		根据计算目标函数采用数据量的不同，梯度下降法又可分为批量梯度下降算法(Batch Gradient Descent)、随机梯度下降算法(Stochastic Gradient Descent)和小批量梯度下降算法(Mini-batch Gradient Descent)

​		假设用只含**一个特征**的线性回归来展开。此时线性回归的**假设函数**为：
$$
h_\theta (x^{(i)}) = \theta_1 x^{(i)} + \theta_0
$$
​		其中$i=1,2,...,m$表示样本数。

​		对应的**目标函数（代价函数）**即为：
$$
J(\theta_0,\theta_1)= \frac{1}{2m} \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2
$$




## 1 批量梯度下降(Batch Gradient Descent, BGD)

特点：使用整个训练集。（又被称为**批量**(batch)或**确定性**(deterministic)梯度算法，因为它们会**在一个大批量中同时处理所有样本**。

1. 对目标函数求偏导：
   $$
   \frac{\Delta J(\theta_0,\theta_1)}{\Delta \theta_j} = \frac{1}{m} \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
   $$
   其中$i=1,2,...,m$表示样本数，$j=0,1$表示特征数，这里使用了偏置项$x_0^{(i)}=1$。

2. 每次迭代对参数进行更新：
   $$
   \theta_j :=\theta_j - \alpha \frac{1}{m} \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
   $$



伪代码：
$$
repeat\{ \\
	\theta_j :=\theta_j - \alpha \frac{1}{m} \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} \\
	(for j=0,1)\\
\}
$$




Advantage:

1. 一次迭代是对所有样本进行计算，可使用矩阵操作，实现并行
2. 由全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。当目标函数为凸函数时，BGD一定能够得到全局最优。



Disadvantage:

1. 当样本数目m很大时，每迭代一步都需要对所有样本计算，训练过程会很慢。





## 2 随机梯度下降(Stochastic Gradient Descent, SGD)

特点：**每次迭代使用一个样本**来对参数进行更新。使得训练速度加快。

对于一个样本的目标函数为：
$$
J^{(i)}(\theta_0,\theta_1) = \frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2
$$

1. 对目标函数求偏导
   $$
   \frac{\Delta J^{(i)}(\theta_0,\theta_1)}{\theta_j}=(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
   $$

2. 参数更新
   $$
   \theta_j := \theta_j - \alpha (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
   $$



伪代码：
$$
repeat\{ \\
for \quad i=1,...,m \{ \\
\theta_j := \theta_j - \alpha (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \\
(for \quad j=0,1)  \\
\} \\
\}
$$


advantage:

1. 不是在全部训练数据上的损失函数，而是在每轮迭代中，随机优化某一条训练数据上的损失函数，这样每一轮参数的更新速度大大加快。



disadvantage:

1. 准确度下降。即使目标函数为强凸函数，SGD仍无法做到线性收敛。
2. 可能会收敛到局部最优，但单个样本并不能代表全体样本的趋势。
3. 不易于并行实现。



## 3 小批量梯度下降(Mini-Batch Gradient Descent, MBGD)

上述两种方法的折中办法。

特点：每次迭代使用 **batch_size** 个样本对参数进行更新。



假设 batch_size=10, 样本数m=1000



伪代码：
$$
repeat\{ \\
	for i = 1,11,21,31,...,991\{ \\
	\theta_j := \theta_j -\alpha \frac{1}{10} \sum_{k=i}^{(i+9)}(h_\theta(x^{(k)})-y^{(k)})x_j^{(k)} \\
	(for j=0,1)	\\
	\}	\\
\}
$$


advantage:

1. 通过矩阵运算，每次在一个batch上优化神经网络参数并不会比单个数据慢太多。
2. 使用batch减少收敛所需迭代次数。
3. 可实现并行化



disadvantage:

1. batch_size选择不当可能会带来问题。



