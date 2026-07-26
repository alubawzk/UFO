# BFM-Zero：FB Representations 与 FB-CPR 公式解释

本文整理并解释 `BFM-Zero.pdf` 第 5 页中以 **“FB representations and FB-CPR.”** 开头的段落。

这篇论文属于无监督强化学习的人形机器人预训练方法。它希望在不预先指定“走路、转身、跟踪某段动作”等具体奖励的情况下，先学习一个通用任务空间，之后通过不同的 latent $z$ 调用不同技能。

## 1. 一句话理解

FB 的核心思想是：

> 用两个网络 $F$ 和 $B$，把“从当前状态动作出发，未来长期会到达哪些状态”压缩成两个低维向量的点积。

其中：

- $F$：描述“从这里出发，未来会往哪里走”；
- $B$：描述“某个未来状态是什么样、适合被哪些轨迹到达”；
- $z$：选择机器人现在要执行哪一种任务或技能；
- Actor $\pi_z$：根据 $z$ 执行对应行为。

## 2. 公式中的符号

论文定义 Forward mapping：

$$
F:\mathcal S\times\mathcal A\times\mathbb R^d\rightarrow\mathbb R^d
$$

$F$ 接收：

```text
当前状态 s
当前动作 a
任务向量 z
```

输出一个 $d$ 维向量：

$$
F(s,a,z)\in\mathbb R^d
$$

Backward mapping 为：

$$
B:\mathcal S\rightarrow\mathbb R^d
$$

它接收一个候选未来状态 $s'$，输出：

$$
B(s')\in\mathbb R^d
$$

其他符号如下：

| 符号 | 含义 |
|---|---|
| $s$ | 当前状态 |
| $a$ | 当前动作 |
| $s'$ | 某个候选未来状态 |
| $z$ | latent task，即任务向量 |
| $\pi_z$ | 由 $z$ 条件控制的策略 |
| $d$ | latent 空间维度，BFM-Zero 中是 256 |
| $\gamma$ | 折扣系数，越远的未来权重越小 |
| $\rho$ | 训练数据中的状态分布 |
| $M^{\pi_z}$ | 在策略 $\pi_z$ 下，未来长期访问状态的统计量 |

## 3. 核心公式 2.1

论文给出的分解为：

$$
M^{\pi_z}(ds'|s,a)
\simeq
F(s,a,z)^\top B(s')\rho(ds')
$$

先忽略 $\rho(ds')$，主要看：

$$
F(s,a,z)^\top B(s')
$$

这是两个 $d$ 维向量的点积，结果是一个标量。

它表示：

> 从当前 $(s,a)$ 出发，之后一直采用任务策略 $\pi_z$，长期访问未来状态 $s'$ 的程度，可以用 $F(s,a,z)$ 和 $B(s')$ 的相似程度来近似。

如果两个向量方向比较一致：

$$
F(s,a,z)^\top B(s')\quad\text{较大}
$$

表示策略未来更倾向于访问这种状态。如果点积较小，则表示这种未来状态与当前策略的长期运动方向不匹配。

可以将其简化为：

```text
F(s,a,z)：未来轨迹的“查询向量”
B(s')：未来状态的“特征向量”

两者点积：未来访问这个状态的分数
```

$\rho(ds')$ 表示训练状态分布对候选状态 $s'$ 的基础权重。

## 4. $M^{\pi_z}$ 到底是什么

论文进一步定义，对于状态空间中的某个区域 $X$：

$$
M^{\pi_z}(s'\in X|s,a)
=
\sum_t \gamma^t
\Pr(s_t\in X|s,a,\pi_z)
$$

它不是普通的一步状态转移概率，而是把所有未来时刻的访问概率加起来：

$$
\Pr(s_0\in X)
+
\gamma\Pr(s_1\in X)
+
\gamma^2\Pr(s_2\in X)
+\cdots
$$

因此：

- 近期进入区域 $X$：权重大；
- 很久以后进入区域 $X$：权重小；
- 多次或长期处于区域 $X$：累计值更大。

所以 $M^{\pi_z}$ 描述的是：

> 从当前状态动作出发，执行策略 $\pi_z$ 后的长期折扣状态访问分布。

这就是论文所说的 long-term policy dynamics。

## 5. 为什么要拆成 $F$ 和 $B$

完整记录所有：

```text
当前状态 × 当前动作 × 任务 z × 所有未来状态
```

之间的长期访问关系非常庞大。FB 使用有限维向量进行近似：

$$
\text{庞大的长期状态转移关系}
\quad\Longrightarrow\quad
F(s,a,z)^\top B(s')
$$

论文将其称为 finite-rank approximation，即有限秩近似。

论文说 $B$ 捕捉 “low-frequency features”。这里不是指机器人控制的 20 Hz 或 50 Hz，而是指能够概括长时间状态依赖关系的特征，而不是只反映瞬时变化。

## 6. 从 $B(s)$ 得到任务特征 $\phi(s)$

论文定义：

$$
\phi(s)
=
\left(
\mathbb E_\rho[B(s)B(s)^\top]
\right)^{-1}
B(s)
$$

可以分两步理解。

首先统计 $B$ 在训练状态分布中的二阶矩阵：

$$
C_B=\mathbb E_\rho[B(s)B(s)^\top]
$$

然后对 $B(s)$ 做变换：

$$
\phi(s)=C_B^{-1}B(s)
$$

这里的 $\phi(s)$ 就是论文所说的 latent task feature，即状态 $s$ 在任务空间中的特征。

两者关系为：

```text
B(s)：FB 分解中学习到的状态表示
          ↓ 经过二阶矩阵变换
φ(s)：用于定义任务和奖励的状态特征
```

## 7. latent $z$ 如何定义任务

论文利用 $\phi(s)$ 和 $z$ 定义奖励：

$$
r_z(s)=\phi(s)^\top z
$$

也就是：

```text
状态特征 φ(s)
      ·
任务方向 z
      ↓
当前状态的任务奖励
```

如果某个状态的 $\phi(s)$ 与 $z$ 方向一致，则：

$$
\phi(s)^\top z
$$

会比较大，说明这个状态符合任务 $z$。

因此，$z$ 不是一个直接测量到的物理量，而是：

> 学习到的任务特征空间中的一个方向，用来指定策略应当偏好哪些状态。

每个 $z$ 都对应：

- 一个奖励函数 $r_z$；
- 一个策略 $\pi_z$；
- 一种由该策略优化的行为。

## 8. 为什么 $F^\top z$ 是 Q-value

论文写道：

$$
\mathbb E\left[
\sum_t\gamma^t\phi(s_t)^\top z
\mid\pi_z
\right]
=
F(s,a,z)^\top z
$$

左边是未来累计奖励：

$$
\sum_t\gamma^t r_z(s_t)
=
\sum_t\gamma^t\phi(s_t)^\top z
$$

而 $F$ 表示未来累计的状态特征，因此再与任务方向 $z$ 点积：

$$
F(s,a,z)^\top z
$$

就得到执行动作 $a$ 后的预期累计任务奖励。

所以：

$$
Q^{\pi_z}(s,a)
=
F(s,a,z)^\top z
$$

也就是：

> $F$ 不只表示未来会访问什么状态；给定任务向量 $z$ 后，它还能直接计算当前动作对该任务的长期价值。

Actor $\pi_z$ 选择能够让这个值尽可能大的动作。

## 9. FB-CPR 比普通 FB 多了什么

普通 FB 通过无奖励交互学习：

- 状态表示 $B$；
- 任务特征 $\phi$；
- successor feature $F$；
- latent-conditioned policy $\pi_z$。

BFM-Zero 使用的 FB-CPR 还加入了一个：

```text
latent-conditioned discriminator
```

判别器接收行为状态和对应的 $z$，用动作数据集 $\mathcal M$ 约束策略，使无监督学习得到的 policy 接近数据集里的示范行为。

因此两者的关系是：

```text
FB
├── 学习长期状态动力学
├── 学习 latent task 空间
└── 学习 π_z 对应的多种策略

FB-CPR
├── 保留以上所有内容
└── 加入判别器，把策略约束到示范动作分布附近
```

论文还强调，FB-CPR 是在线、off-policy 训练的，不要求离线数据集覆盖所有可能的状态和行为。

## 10. 整体逻辑

这一段的完整逻辑可以压缩为：

```text
机器人在线探索
        ↓
B(s') 编码候选未来状态
        ↓
F(s,a,z) 编码当前动作在任务 z 下的长期未来
        ↓
FᵀB 近似长期状态访问分布
        ↓
由 B 推导任务特征 φ
        ↓
r_z(s)=φ(s)ᵀz 定义任务奖励
        ↓
F(s,a,z)ᵀz 成为该任务的 Q-value
        ↓
Actor π_z 选择使 Q-value 最大的动作
        ↓
FB-CPR 判别器进一步约束动作接近示范数据
```

最需要记住的三个关系是：

$$
\boxed{F^\top B=\text{未来会访问哪些状态}}
$$

$$
\boxed{\phi^\top z=\text{当前状态对任务 }z\text{ 的奖励}}
$$

$$
\boxed{F^\top z=\text{当前动作对任务 }z\text{ 的长期价值}}
$$

## 11. 与后续 reference motion latent 的关系

这一段只把 $z$ 定义为抽象任务向量，尚未定义某段具体 reference motion 如何转换成 $z$。

论文在后续预训练部分给出了动作轨迹 $\tau$ 对应的 zero-shot imitation embedding：

$$
z_\tau
=
\frac{1}{l(\tau)}
\sum_{(o,s)\in\tau}B(o,s)
$$

即对轨迹中的状态分别计算 $B$，再在时间维度上进行平均，得到代表整段动作的 latent $z_\tau$。

