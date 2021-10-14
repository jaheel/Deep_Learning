# BN

batch normalization

批规范化

针对单个样本的output值的规范化



论文里的公式：
$$
Value \space of \space x \space over \space a \space mini-batch : B=\{ x_1,\dots,x_m \} \\
hyper \space parameter: \gamma,\beta \\
\space\\
\mu_B \leftarrow \frac{1}{m} \sum_{i=1}^m x_i \space \space //mini-batch \space mean \\
\sigma_B^2 \leftarrow \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2 \space \space //mini-batch \space variance \\
\hat{x_i} \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \space \space //normalize \\
y_i \leftarrow \gamma \hat{x_i} + \beta \equiv BN_{\gamma,\beta}(x_i) \space // scale \space and \space shift
$$


feature map(N个样本，通道数为C，高为H，宽为W)
$$
x \in R^{N \times C \times H \times W}
$$
具体公式：
$$
\mu_c(x) = \frac{1}{NHW} \sum_{n=1}^N \sum_{h=1}^H \sum_{w=1}^W x_{nchw} \\
\sigma_c(x) = \sqrt{ \frac{1}{NHW} \sum_{n=1}^N \sum_{h=1}^H \sum_{w=1}^W(x_{nchw} - \mu_c(x))^2 + \epsilon} \\
x_{out} = \gamma(\frac{x - \mu_c(x)}{\sigma_c(x)}) + \beta
$$



# reference

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

[深入理解Batch Normalization](https://zhuanlan.zhihu.com/p/87117010)