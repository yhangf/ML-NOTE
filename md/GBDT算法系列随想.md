# <center>$GBDT $算法系列随想</center>

<center><strong>杨航锋</strong></center>

#### 1 分裂点寻找与优化

令输入空间是$\ X\in \mathbb{R}^{n} \ $,输出空间是$\ Y\in\mathbb{R}\ $，不妨假设含有$m$个样本数据($x^{(1)}$,$y^{(1)}$)、($x^{(2)}$,$y^{(2)}$)、$\cdots$、($x^{(m)}$,$y^{(m)}$)，其中$x^{(i)}\in X、y^{(i)}\in Y \ $。$\ T\ $为叶子结点个数，$\ \zeta $为输出数据$y^{(i)} \ $在叶子结点$\ T\ $上的划分，$\ |\zeta|\ $表示在该划分下输出数据$\ y^{(i)}\ $的个数，因此可以定义$\ GBDT\ $回归树的整体$\ loss \ $
$$
L^{(before)}=\sum\limits_{j=1}^T\sum\limits_{i\in\zeta_j}(y^{(i)}-\frac{1}{|\zeta_j|}\sum\limits_{i\in\zeta_j}y^{(i)})^2
$$


如果回归树还能继续分裂下去，不妨假设落在第$\ \color{red} j\ $个结点的数据，随着特征点$\ x_k^{(i)} \ $的选择后继续分裂成两部分，一部分是左子结点部分$\ \zeta_j^{(l)}\ $另一部分为右子结点$\ \zeta_j^{(r)} \ $，那么分裂后的$\ loss \ $就转化为
$$
L^{(end)}=\sum\limits_{t=1,j\neq t}^T\sum\limits_{i\in\zeta_j}(y^{(i)}-\frac{1}{|\zeta_j|}\sum\limits_{i\in\zeta_j}y^{(i)})^2+\sum\limits_{i\in\zeta_j^{(l)}}(y^{(i)}-\frac{1}{|\zeta_j^{(l)}|}\sum\limits_{i\in\zeta_j^{(l)}}y^{(i)})^2+\sum\limits_{i\in\zeta_j^{(r)}}(y^{(i)}-\frac{1}{|\zeta_j^{(r)}|}\sum\limits_{i\in\zeta_j^{(r)}}y^{(i)})^2
$$
最后希望分裂之后的$\ loss\ $比分裂之前的$\ loss\ $要小，当然越小越好，即求解
$$
\begin{aligned}
\mathop{\arg\max}_{x_k^{(i)}}\ \Delta L&=L^{(before)}-L^{(end)}\\
&=\sum\limits_{i\in\zeta_j}(y^{(i)}-\frac{1}{|\zeta_j|}\sum\limits_{i\in\zeta_j}y^{(i)})^2-\sum\limits_{i\in\zeta_j^{(l)}}(y^{(i)}-\frac{1}{|\zeta_j^{(l)}|}\sum\limits_{i\in\zeta_j^{(l)}}y^{(i)})^2-\sum\limits_{i\in\zeta_j^{(r)}}(y^{(i)}-\frac{1}{|\zeta_j^{(r)}|}\sum\limits_{i\in\zeta_j^{(r)}}y^{(i)})^2
\end{aligned}
$$
对于上式的求解问题，可以通过遍历$\ x^{(i)}\ $中的所有取值，然后筛选出使得$\ \Delta L \ $最大的切分点$\ x_k^{(i)} \ $即可，这样做的算法时间复杂度为$\ \Theta(n^3)\ $显然这并不是最优解。那么该如何优化呢？由于$\ \sum\limits_{i\in\zeta_j}(y^{(i)}-\dfrac{1}{|\zeta_j|}\sum\limits_{i\in\zeta_j}y^{(i)})^2\ $为固定值且与待优化参数没有关系，故优化目标可以改写成
$$
\begin{aligned}
\mathop{\arg\min}_{x_k^{(i)}}\ \sum\limits_{i\in\zeta_j^{(l)}}(y^{(i)}-\frac{1}{|\zeta_j^{(l)}|}\sum\limits_{i\in\zeta_j^{(l)}}y^{(i)})^2+\sum\limits_{i\in\zeta_j^{(r)}}(y^{(i)}-\frac{1}{|\zeta_j^{(r)}|}\sum\limits_{i\in\zeta_j^{(r)}}y^{(i)})^2
\end{aligned}
$$
求解这个问题之前先看一个引理

> $$
> \begin{aligned}
> \sum\limits_{i=1}^n(x_i-\overline{x})^2&=\sum\limits_{i=1}^n(x_i^2-2\overline{x}x_i+\overline{x}^2)\\
> &=\sum\limits_{i=1}^n x_i^2-2\overline{x} \cdot n \cdot \frac{1}{n}\sum\limits_{i=1}^n x_i+ n\overline{x}^2\\
> &=\sum\limits_{i=1}^n x_i^2-n\overline{x}^2
> 
> 
> \end{aligned}
> $$

所以可以把优化目标化简为
$$
\begin{aligned}
\mathop{\arg\min}_{x_k^{(i)}}&\ \sum\limits_{i\in\zeta_j^{(l)}}(y^{(i)}-\frac{1}{|\zeta_j^{(l)}|}\sum\limits_{i\in\zeta_j^{(l)}}y^{(i)})^2+\sum\limits_{i\in\zeta_j^{(r)}}(y^{(i)}-\frac{1}{|\zeta_j^{(r)}|}\sum\limits_{i\in\zeta_j^{(r)}}y^{(i)})^2\\
&=\sum\limits_{i\in\zeta_j^{(l)}}(y^{(i)})^2+\sum\limits_{i\in\zeta_j^{(r)}}(y^{(i)})^2
-\frac{1}{|\zeta_j^{(l)}|}(\sum\limits_{i\in\zeta_j^{(l)}}y^{(i)})^2-\frac{1}{|\zeta_j^{(r)}|}(\sum\limits_{i\in\zeta_j^{(r)}}y^{(i)})^2\\
&=\sum\limits_{i}(y^{(i)})^2
-\frac{1}{|\zeta_j^{(l)}|}(\sum\limits_{i\in\zeta_j^{(l)}}y^{(i)})^2-\frac{1}{|\zeta_j^{(r)}|}(\sum\limits_{i\in\zeta_j^{(r)}}y^{(i)})^2
\end{aligned}
$$
由于$\ \sum\limits_{i}(y^{(i)})^2 \ $是定值那么最后的优化求解问题转化为，确定切分点$\ x_k^{(i)} \ $使得$\frac{1}{|\zeta_j^{(l)}|}(\sum\limits_{i\in\zeta_j^{(l)}}y^{(i)})^2+\frac{1}{|\zeta_j^{(r)}|}(\sum\limits_{i\in\zeta_j^{(r)}}y^{(i)})^2 \ $的最大值问题，算法时间复杂度为$\ \Theta(n^2)\ $。

#### 2