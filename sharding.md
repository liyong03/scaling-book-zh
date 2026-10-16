---
layout: distill
title: "分片矩阵及其乘法"
# permalink: /main/
description: "训练大型机器学习模型时，我们必须把模型的参数或输入切分（即“分片”）到许多加速器上。由于 LLM 主要由矩阵乘法构成，理解这一点归根结底就是理解当矩阵被切分到多个设备上时该如何相乘。我们基于 TPU 通信原语的开销，建立了一套关于分片矩阵乘法的简单理论。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 3

previous_section_url: "../tpus"
previous_section_name: "第 2 部分：TPU"

next_section_url: ../transformers
next_section_name: "第 4 部分：Transformer 数学"

giscus_comments: false
giscus_discussion: 5

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "分区记号与集合通信操作"
    id: partitioning-notation-and-collective-operations
  - subsections:
    - name: "统一的分片记号"
      id: a-unified-notation-for-sharding
    - name: "如何在代码中描述分片？"
      id: how-do-we-describe-this-in-code
  - name: "使用分片数组进行计算"
    id: computation-with-sharded-arrays
  - subsections:
    - name: "情形 1：两个乘数的收缩维度都未分片"
      id: case-1-neither-multiplicand-has-a-sharded-contracting-dimension
    - name: "情形 2：一个乘数的收缩维度已分片"
      id: case-2-one-multiplicand-has-a-sharded-contracting-dimension
    - name: "情形 3：两个乘数的收缩维度都已分片"
      id: case-3-both-multiplicands-have-sharded-contracting-dimensions
    - name: "情形 4：两个乘数各有一个非收缩维度沿同一轴分片"
      id: case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis
  - name: "深入理解 TPU 通信原语"
    id: a-deeper-dive-into-tpu-communication-primitives
  - subsections:
    - name: "最后一个通信原语：AllToAll"
      id: our-final-communication-primitive-the-alltoall
    - name: "再谈 ReduceScatter"
      id: more-about-the-reducescatter
    - name: "如何让矩阵乘法的通信与计算重叠"
      id: how-to-overlap-matmul-communication-with-compute
  - name: "我们学到了什么？"
    id: what-have-we-learned
  - name: "练习题"
    id: some-problems-to-work

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## 分区记号与集合通信操作 {#partitioning-notation-and-collective-operations}

当我们在一万块 TPU 或 GPU 上训练 LLM 时，抽象来看，我们做的计算与在单块芯片上训练并无二致。区别在于，**我们的数组装不进单块 TPU/GPU 的 HBM**，因此必须把它们切分开。<d-footnote>值得一提的是，我们也可能出于速度考虑而选择并行化。即使模型能放进较少的芯片，扩展到更多芯片也能直接带来更多的 FLOPs/s。例如在推理时，我们有时能放进较小的拓扑，却选择扩展到更大的拓扑以降低延迟。同样，在训练时我们也常常扩展到更多芯片来缩短每一步的时间。</d-footnote>我们把这称为对数组进行“*分片*（sharding）”或“*分区*（partitioning）”。扩展的艺术就在于想清楚如何对模型进行分片，使计算依然保持高效。

下面是一个二维数组 **A** 分片到 4 块 TPU 上的例子：

{% include figure.liquid path="assets/img/sharding-example.png" class="img-fluid" caption="<b>图：</b>一个形状为 <b>A</b>[I, J] 的示例数组被分片到 4 个设备上。两个维度都各自均匀地分片到 2 个设备上，分片方式为 <b>A</b>[I<sub>X</sub>, J<sub>Y</sub>]。每块 TPU 持有总内存的 1/4。" %}

注意，分片后的数组仍然具有与未分片数组相同的*全局*形状（或称*逻辑形状*），比如 `(4, 128)`；但它还有一个*设备本地形状*，比如 `(2, 64)`，它给出了每块 TPU 实际持有的字节大小（在上图中，每块 TPU 持有整个数组的 ¼）。下面我们把这推广到任意数组。

### 统一的分片记号 {#a-unified-notation-for-sharding}

我们使用一种*命名轴记号*（named-axis notation）的变体来描述张量*如何*以块的形式分片到各个设备上：假设存在一个二维或三维的设备网格，称为**设备网格（device mesh）**，其中每个轴都被赋予了**网格轴名称**，**例如 X**、**Y 和 Z。** 然后，我们可以通过描述数组的每个命名维度如何在物理网格轴上分区，来指定矩阵数据在设备网格上的布局方式。我们把这种指派称为一个**分片**（sharding）。

**示例（上图）**：对于上图，我们有：
* **网格：** 上面的设备网格 `Mesh(devices=((0, 1), (2, 3)), axis_names=('X', 'Y'))`，它告诉我们有 4 块 TPU 排成 2x2 的网格，轴名称为 $X$ 和 $Y$。
* **分片：** $A[I_X, J_Y]$，它告诉我们将第一个轴 $I$ 沿网格轴 $X$ 分片，将第二个轴 $J$ 沿网格轴 $Y$ 分片。这个分片告诉我们，每个分片持有数组的 $1 / (\lvert X\rvert \cdot \lvert Y\rvert)$。

综合起来，我们知道数组的本地形状（单个设备持有的分片大小）为 $(\lvert I\rvert / 2, \lvert J\rvert / 2)$，其中 $$\lvert I\rvert$$ 是 A 第一个维度的大小，$$\lvert J\rvert$$ 是 A 第二个维度的大小。

<b markdown=1 style="color: #048affff;">小测验 [在 1 个轴上做二维分片]：</b>考虑一个数组 `fp32[1024, 4096]`，其分片为 $A[I_{XY}, J]$，网格为 `{'X': 8, 'Y': 2}`。每个设备持有多少数据？在 H100 上从 HBM 加载这个数组需要多长时间（假设每块芯片的内存带宽为 `3.4e12`）？

{% details 点击此处查看答案。 %}

$A[I_{XY}, J]$ 将第一个维度（I）同时沿 X 和 Y 两个硬件轴分片。在这个例子中，本地形状为 $(\lvert I\rvert /(\lvert X\rvert \cdot \lvert Y\rvert), \lvert J\rvert)$。对于给定的例子，全局形状是 `fp32[1024, 4096]`，因此本地形状是 `fp32[64, 4096]`。

由于每块 GPU 持有 `4 * 64 * 4096 = 1MiB` 字节，这大约需要 `1e6 / 3.4e12 = 294ns`，不过由于数据量太小，各种开销很可能会让实际时间明显更长。

{% enddetails %}

**可视化这些分片：** 我们来看一个切分到 4 个设备上的二维数据数组，试着把这些分片可视化：

{% include figure.liquid path="assets/img/sharding-colored1.png" class="img-fluid img-small" %}

我们把矩阵的*完全复制*（fully-replicated）形式简单地写作 $A[I, J]$，不带任何分片指派。这意味着*每个*设备都持有整个矩阵的完整副本。

{% include figure.liquid path="assets/img/sharding-colored2.png" class="img-fluid img-small" %}

我们可以用网格轴下标来表示某个维度已沿某个网格轴分区。例如 $A[I_X, J]$ 表示逻辑轴 **I** 已沿网格维度 **X** 分区，而维度 **J** *没有*分区，并且这些块在网格轴 **Y** 上保持*部分复制*（partially-replicated）。

{% include figure.liquid path="assets/img/sharding-colored3.png" class="img-fluid img-small" %}

$A[I_X, J_Y]$ 表示逻辑轴 **I** 已沿网格轴 **X** 分区，并且维度 **J** 已沿网格轴 **Y** 分区。

{% include figure.liquid path="assets/img/sharding-colored4.png" class="img-fluid img-small" %}

下图展示了其他几种可能的情况：

{% include figure.liquid path="assets/img/sharding-colored5.png" class="img-fluid" %}

这里 $A[I_{XY}, J]$ 表示我们把网格轴 **X** 和 **Y** 视为一个展平后的更大维度，并把命名轴 **I** 分区到所有设备上。多个网格轴下标的顺序很重要，因为它规定了分区在网格上的遍历顺序。

{% include figure.liquid path="assets/img/sharding-colored6.png" class="img-fluid img-small" %}

最后要注意，我们*不能*让多个命名轴沿*同一个*网格维度分片。例如 $A[I_X, J_X]$ 就是一个无意义的、被禁止的分片。一旦某个网格维度已被用于对数组的某个维度分片，它在某种意义上就已经被“用掉”了。

<b markdown=1 style="color: #57cf57;">小测验：</b>设 **A** 是一个形状为 `int8[128, 2048]` 的数组，分片为 $A[I_{XY}, J]$，网格为 `Mesh({'X': 2, 'Y': 8, 'Z': 2})`（因此总共 32 个设备）。**A** 在每个设备上占用多少内存？**A** 在所有设备上总共占用多少内存？

{% details 点击此处查看答案。 %}

**答案：** 我们的数组 **A** 沿 X 和 Y 分片、沿 Z 复制，因此每个设备上的形状为 `int8[128 / (2 * 8), 2048] = int8[8, 2048]`，大小为 `8 * 2048 = 16,384` 字节。由于它沿 Z 复制，而在每个 Z 平面内沿 X 和 Y 完全分片，所以原数组共有 2 份完整副本（每个 Z 平面一份）。因此所有设备上的总大小为：原数组大小 × Z 方向副本数 = 128 * 2048 * 2 = 512 KiB。也可以这样验证：32 个设备 × 每个设备 16,384 字节 = 总共 512 KiB。

{% enddetails %}

### 如何在代码中描述分片？ {#how-do-we-describe-this-in-code}

到目前为止我们一直没谈代码，但现在正好可以先睹为快。JAX 使用一种命名分片语法，它与我们上面描述的抽象语法非常接近。我们会在[第 10 部分](../jax-stuff)中详细讨论，这里先简单预览一下。你可以在[这里](https://colab.research.google.com/drive/15cxw66eABwZPG-V4QFmbLfiykPFf_gaP?usp=sharing)的 Google Colab 中动手试试，并对结果做性能分析，看看 JAX 如何处理不同的分片。这段代码做了 3 件事：

1. 创建一个 **jax.Mesh**，把我们的 8 块 TPU 映射成 4x2 的网格，两个轴分别命名为 'X' 和 'Y'。
2. 创建矩阵 A 和 B，其中 A 在两个维度上都分片，B 沿输出维度分片。
3. 编译并执行一个简单的矩阵乘法，返回一个分片数组。

```py
import jax
import jax.numpy as jnp

# 创建我们的 mesh！我们运行在一个 TPU v2-8 4x2 切片上，轴名为 'X' 和 'Y'。
# Auto 轴类型告诉 JAX 让 XLA 编译器推断中间结果的分片。
assert len(jax.devices()) == 8
Auto = jax.sharding.AxisType.Auto
mesh = jax.make_mesh(axis_sizes=(4, 2), axis_names=('X', 'Y'), axis_types=(Auto, Auto))

# 一个帮助定义分片的小工具函数。PartitionSpec 就是我们的
# 分片方式（从轴到名称的映射）。
def P(*args):
  return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# 我们将 A 和 B 都沿非收缩维度分片，并将 A 沿收缩维度分片。
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# 我们可以对这些分片数组执行 matmul！out_shardings 指定了我们希望
# 输出如何分片。其余的分片工作由 JAX/XLA 替我们处理。
y = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y'))(A, B)
```

JAX 的妙处在于，这些数组用起来就像没有分片一样！`B.shape` 会告诉我们全局形状（或逻辑形状）(2048, 8192)。我们必须查看 `B.addressable_shards` 才能看到它在本地是如何分片的。我们可以对这些数组执行各种操作，JAX 会设法弄清如何对它们进行广播或重塑来完成这些操作。例如在上面的例子中，**A** 的本地形状是 `[2, 1024]`，**B** 的本地形状是 `[2048, 4096]`。JAX/XLA 会根据需要自动在这些数组之间加入通信，以完成最终的乘法。

## 使用分片数组进行计算 {#computation-with-sharded-arrays}

如果你有一个分布在许多设备上的数据数组，并希望对它执行数学运算，那么同时对数据和计算进行分片会带来哪些开销？

显然，这取决于具体涉及的计算。

* 对于*逐元素*（elementwise）运算，在分布式数组上操作**没有任何开销**。
* 当我们希望对分布在许多设备上的元素进行跨设备运算时，事情就变复杂了。幸运的是，在大多数机器学习中，几乎所有计算都以矩阵乘法的形式进行，而矩阵乘法相对容易分析。

本节余下部分将讨论如何对分片矩阵做乘法。粗略来说，这需要把矩阵的各个块搬来搬去，使得每个块都能被完整地相乘或求和。**每种分片方式都会涉及不同的通信。** 例如，$A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$ 无需任何通信就能完成乘法，因为*收缩维度*（contracting dimension，即 J，也就是我们实际求和的那个维度）没有分片。然而，如果我们希望输出不分片（即 $A[I_X, J] \cdot B[J, K_Y] \to C[I, K]$），就需要把 $A$ 和 $B$，或者把 $C$ 复制到每个设备上（使用 *AllGather*）。这两种选择的通信开销不同，所以我们需要计算各自的开销并选择最低的那个。

{% details 你可以从“分块矩阵乘法”的角度来理解这一点。 %}

为了理解这一点，回顾一下“分块矩阵”的概念会有所帮助，也就是由矩阵嵌套构成的矩阵：

$$\begin{equation}
\begin{pmatrix}
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\
a_{20} & a_{21} & a_{22} & a_{23} \\
a_{30} & a_{31} & a_{32} & a_{33}
\end{pmatrix}
=
\left(
\begin{matrix}
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix} \\
\begin{bmatrix}
a_{20} & a_{21} \\
a_{30} & a_{31}
\end{bmatrix}
\end{matrix}
\begin{matrix}
\begin{bmatrix}
a_{02} & a_{03} \\
a_{12} & a_{13}
\end{bmatrix} \\
\begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
\end{matrix}
\right)
=
\begin{pmatrix}
\mathbf{A_{00}} & \mathbf{A_{01}} \\
\mathbf{A_{10}} & \mathbf{A_{11}}
\end{pmatrix}
\end{equation}$$

矩阵乘法有一个很好的性质：当乘数矩阵按块写出时，乘积可以按照标准规则写成若干块矩阵乘法的组合：

$$\begin{equation}
\begin{pmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{pmatrix}
\cdot
\begin{pmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{pmatrix}
=
\begin{pmatrix}
A_{00}B_{00} + A_{01}B_{10} & A_{00}B_{01} + A_{01}B_{11} \\
A_{10}B_{00} + A_{11}B_{10} & A_{10}B_{01} + A_{11}B_{11}
\end{pmatrix}
\end{equation}$$

这意味着，实现分布式矩阵乘法可以归结为：在网络上搬运这些分片块，对这些块执行*本地*矩阵乘法，再把结果求和。**于是问题就变成了：需要加入什么样的通信，以及这些通信有多昂贵。**

{% enddetails %}

方便的是，我们可以把所有可能的分片方式大致归纳为需要考虑的 4 种情形，每种情形都有一条规则说明需要加入什么通信
1. **[情形 1](#case-1-neither-multiplicand-has-a-sharded-contracting-dimension)：** 两个输入都没有沿收缩维度分片。*我们可以直接将本地分片相乘，无需任何通信。*
2. **[情形 2](#case-2-one-multiplicand-has-a-sharded-contracting-dimension)：** 一个输入的收缩维度已分片。*我们通常沿收缩维度对已分片的输入做“AllGather”。*
3. **[情形 3](#case-3-both-multiplicands-have-sharded-contracting-dimensions)：** 两个输入都沿收缩维度分片。*我们可以先将本地分片相乘，再对结果做“AllReduce”。*
4. **[情形 4](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)：** 两个输入各有一个非收缩维度沿同一个轴分片。我们必须先对两个输入之一做 AllGather 才能继续。

你可以把这些看作只需遵循的规则，但理解这些规则为何成立以及它们的开销有多大也很有价值。下面我们逐一详细讲解。

### 情形 1：两个乘数的收缩维度都未分片 {#case-1-neither-multiplicand-has-a-sharded-contracting-dimension}

**引理：** 对分片矩阵做乘法时，计算是合法的，且输出沿用输入的分片方式，*除非*收缩维度被分片，或者两个矩阵沿同一个轴分片。例如，下面这样完全没问题

$$\begin{equation*}
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow \mathbf{C}[I_X, K_Y]
\end{equation*}$$

而且完全不需要任何通信，得到的结果张量同时沿 X 和 Y 两个硬件维度分片。试着想一想为什么会这样。基本上，计算与分片方式是*无关*的，因为每个批次条目都持有被收缩轴的某个本地块，可以在本地完成相乘和归约。下面这些情形都没问题，并且都遵循这条规则：

$$\begin{align*}
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I, K] \\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I_X, K]\\
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I, K_Y]\\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I_X, K_Y]
\end{align*}$$

由于 **A** 和 **B** 的收缩维度 **J** 都没有分片，我们只需对输入执行本地的分块矩阵乘法，得到的结果就*已经*按照期望的输出分片方式分好片了。当两个乘数的非收缩维度沿同一个轴分片时，情况就不再如此了（详见[非法分片](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)一节）。

### 情形 2：一个乘数的收缩维度被分片 {#case-2-one-multiplicand-has-a-sharded-contracting-dimension}

我们来考虑这样一种情况：一个输入 **A** 沿收缩维度 **J** 分片，而 **B** 是完全复制的：

$$\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

我们不能简单地把 **A** 和 **B** 的本地块直接相乘，因为我们需要在 **A** 的完整收缩维度上求和，而这个维度被切分到了 X 轴上。通常的做法是先对 **A** 的各个分片做“**AllGather**”，使每个设备都拥有一份完整副本，然后再与 **B** 相乘：

$$\textbf{AllGather}_X[I, J_X] \rightarrow \mathbf{A}[I, J]$$

$$\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

这样，实际的乘法就可以在每个设备上完整地完成。

<p markdown=1 class="takeaway">**要点：**当参与相乘的某个矩阵沿收缩维度分片时，我们通常先对它做 AllGather，使收缩维度不再分片，然后再做本地矩阵乘法。</p>

注意，如果 **B** 并没有同样沿 X 分片，我们也可以先做本地的部分矩阵乘法，再对分片的部分和求和（即做 *AllReduce*）。这样可以把计算也分片，但通常通信开销更高。这在某些情况下可能更快，不过在实践中 **B** 通常也是分片的。[下文](#some-problems-to-work)的问题 4 会详细分析什么时候这种做法更好。

**什么是 AllGather？** AllGather 是我们要讨论的第一个核心 [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 通信原语。AllGather 会*移除*沿某个轴的*分片*，把分散在各个设备上的分片重新组装到该轴上的*每个*设备上。用上面的记号来说，AllGather 会从一组轴中移除一个下标，例如：

$$\textbf{AllGather}_{XY}(A[I_{XY}, J]) \rightarrow A[I, J]$$

我们并不需要移除某个维度的全部下标，例如 $$A[I_{XY}, J] \rightarrow A[I_Y, J]$$ 也是一个 AllGather，只不过只作用于单个轴。另外还要注意，我们也可能需要用 AllGather 来移除*非收缩*维度上的分片，例如在下面的矩阵乘法中：

$$A[I_X, J] \cdot B[J, K] \rightarrow C[I, K]$$

我们既可以一开始就对 **A** 做 AllGather 来移除输入的分片，也可以先做分片的矩阵乘法，再对结果 **C** 做 AllGather。

**AllGather 实际上是如何执行的？** 要在单个 TPU 轴（一个环）上执行一维 AllGather，基本做法是让每个 TPU 把自己的分片沿环传递下去，直到每个设备都拥有一份副本。<d-footnote>GPU 上的 AllGather 也可以这样工作：把一个节点内的 GPU 组成一个环，然后按这个（任意的）顺序传递数据块。</d-footnote>下面是一个动画：

{% include figure.liquid path="assets/img/all-gather.gif" caption="<b>图：</b>演示如何在 8 个 TPU 或 GPU 设备上执行 AllGather 的动画。每个设备一开始持有数组的 1 / 8，最终得到一份完整副本。" %}

AllGather 既可以单向进行，也可以双向进行（上图展示的是双向）。如果单向进行，每个 TPU 沿环发送 $N - 1$ 跳，每跳的数据块大小为 $\text{字节} / N$。如果双向进行，则需要 $\lfloor \frac{N}{2} \rfloor$ 跳，每跳大小为 $2 \cdot \text{字节} / N$。

**这需要多长时间？** 我们以双向 AllGather 为例来计算它的耗时。设 $$V$$ 为数组的字节数，$X$ 为收缩维度上的分片数。那么由上图可知，每一跳在每个方向上发送 $V / \lvert X\rvert$ 字节，因此每一跳耗时

$$T_{hop} = \frac{2 \cdot V}{\lvert X \rvert \cdot W_\text{ici}}$$

其中 $W_\text{ici}$ 是**双向** ICI 带宽。<d-footnote>分子中的系数 2 来自我们使用的是双向带宽。我们在每个方向上发送 $V / X$，总共是 $2V / X$。</d-footnote>我们总共需要 $\lvert X\rvert / 2$ 跳才能到达每个 TPU<d-footnote>严格来说是 $\lfloor X / 2 \rfloor$</d-footnote>，因此整个归约耗时

$$T_{total} = \frac{2 \cdot V \cdot X}{2 \cdot X \cdot W_\text{ici}}$$

$$T_{total} = \frac{V}{W_\text{ici}}$$

注意，这个结果**与 $X$ 无关！**这一点相当惊人，因为它意味着尽管我们的 TPU 只是局部相连，连接的局部性却并不重要。我们只受限于每条链路的速度。

<p markdown=1 class="takeaway">**要点：**在吞吐量受限的情况下执行 AllGather（或 ReduceScatter、AllReduce）时，实际的通信时间只取决于数组的大小和可用带宽，而与数组被分片到多少个设备上无关！</p>

**关于 ICI 延迟的说明：**无论数据量多大，ICI 链路上的每一跳都有一些固有开销，通常约为 1us。这意味着当数组 $$A$$ 非常小、每一跳耗时不到 1us 时，我们可能进入“延迟受限”的状态，此时计算耗时*确实*取决于 $X$。

{% details 点击此处查看完整细节。 %}

设 $$T_\text{min}$$ 为单跳的最短时间。那么

$$T_{hop} = \max \left[ T_{min}, \frac{2 \cdot V}{X \cdot W_\text{ici}} \right]$$

$$T_{total} = \max \left[ \frac{T_{min} \cdot X}{2}, \frac{V}{W_\text{ici}} \right]$$

因为我们执行了 $X / 2$ 跳。对于大规模的归约或收集操作，我们毫无疑问是带宽受限的。我们发送的数据量非常大，每一跳的开销基本可以忽略不计。但对于小数组（例如从模型采样时），这一开销就不可忽略了，而 ICI 带宽也变得无关紧要，我们完全受限于延迟。换一种说法：对于某款特定的 TPU，例如单向 ICI 带宽为 `4.5e10` 的 TPU v5e，发送任何小于 `4.5e10 * 1e-6 = 45kB` 的缓冲区都会是延迟受限的。

{% enddetails %}

下面是在 TPU v5e 8x16 切片上实测的 AllGather 带宽。数组沿大小为 16 的轴分片，因此它拥有一个完整的双向环。

{% include figure.liquid path="assets/img/all-gather-bandwidth.png" class="img-small" caption="<b>图：</b>TPU v5e 在 AllGather 过程中的实测带宽和估算的链路带宽。橙色的 BW 是实际每秒完成 AllGather 的字节数，蓝色曲线则是根据该集合通信的已知开销计算出的实测单向链路带宽。" %}

注意，我们不仅达到了标称峰值带宽（`4.5e10`）的约 95%，而且在约 10MB 时就达到了这一峰值；在 16 路分片下，这相当于每个设备约 625kB（*顺带一提*：这比 GPU 好得多）。

**在多个轴上做 AllGather 会怎样？** 当我们在多个轴上做收集时，就有多个维度的 ICI 可以用来执行收集。例如，AllGather<sub>XY</sub>([B, D<sub>XY</sub>]) 作用于两个硬件网格轴。这会使可用带宽增加 $N_\text{axes}$ 倍。

把延迟也考虑进来，我们得到一般规律：

$$T_{total} = \max \left[ \frac{T_{min} \cdot \sum_{i} |X_i|}{2}, \frac{V}{W_\text{ici} \cdot N_\text{axes}} \right]$$

其中 $$\sum_i \lvert X_i \rvert / 2$$ 是 TPU 网格中最长路径的长度。

<b markdown=1 style="color:rgb(144, 92, 255);">小测验 2 [AllGather 耗时]：</b>利用[第 2 部分](../tpus)中的数据，在具有二维网格 `{'X': 8, 'Y': 4}` 的 TPU v5e 上，以 bfloat16 精度、$$E = 2048$$、$$F = 8192$$ 执行 AllGather<sub>Y</sub>([E<sub>Y</sub>, F]) → [E, F] 需要多长时间？如果 $$E=256, F=256$$ 呢？

{% details 点击此处查看答案。 %}

**答案：**我们先计算一些基本量：

1) TPU v5e 的 2 个轴每个都有 4.5e10 bytes/s 的单向 ICI 带宽。
2) 对于 (a)，在 bfloat16 下我们有 $A[E_Y, F]$，因此每个设备持有一个形状为 bf16[512, 8192] 的数组，大小为 512 * 8192 * 2 = 8.4MB。整个数组的大小为 2048 * 8192 * 2 = 34MB。

*对于第 (1) 问*，我们可以使用上面的公式。由于是在一个轴上执行 AllGather，我们有 $T_{\text{通信}} = \text{34e6} / \text{9e10} = \text{377us}$。为了确认我们不是延迟受限的：在大小为 4 的轴上最多只有 3 跳，所以延迟下限大约是 3us，远未接近。然而，TPU v5e 只有在某个轴的大小为 16 时才有环绕连接，所以这里*我们实际上无法做完全双向的 AllGather*。来自一端的数据需要 3 跳才能到达另一端，因此理论上耗时更接近 $T_{\text{通信}} = 3 * \text{8.4e6} / \text{4.5e10} = 560\mu s$。[**这是**](https://imgur.com/a/RkvpRGQ)来自[这个 Colab](https://colab.research.google.com/drive/15tDZMfNqm2vJjvSzw5VC9qtSwc5td-oV?usp=sharing) 的**一份实际性能分析结果（profile）**，显示耗时为 $680 \mu s$，这是合理的，因为我们很可能达不到 100% 的理论带宽！*对于第 (2) 问*，每个分片的大小为 `64 * 256 * 2 = 32kB. 32e3 / 4.5e10 = 0.7us`，所以我们是延迟受限的。由于有 3 跳，这大约需要 3 * 1us = 3us。[实际上更接近 8us。](https://imgur.com/a/HZLQmYs)

{% enddetails %}

<p markdown=1 class="takeaway">**注意：**当我们有一个像 `{'X': 16, 'Y': 4}` 这样的二维网格时，并不要求每个轴都对应某个特定的*硬件*轴。这意味着，例如上面的网格可以描述一个 4x4x4 的 TPU v5p 立方体，其中 $X$ 轴占用了 2 个硬件轴。后面我们描述在多个轴上做数据并行时会用到这一点。</p>

### 情形 3：两个乘数的收缩维度都被分片 {#case-3-both-multiplicands-have-sharded-contracting-dimensions}

第三种基本情形是：两个乘数都在各自的收缩维度上、沿同一个网格轴分片：

$$\textbf{A}[I, J_X] \cdot \textbf{B}[J_X, K] \rightarrow C[I, K]$$

在这种情况下，*本地*的分片分块矩阵乘法至少是*可以*执行的，因为它们共享相同的收缩索引集合。但每个乘积只代表完整目标乘积的一个*部分和*，沿 **X** 维度的每个设备最终会持有这个最终乘积的不同*部分和*。这种情况非常常见，因此我们扩展记号来显式标记它：

$$\textbf{A}[I, J_X] \cdot_\text{LOCAL} \textbf{B}[J_X, K] \rightarrow C[I, K] \{\ U_X \}$$

记号 **{ U<sub>X</sub> }** 读作“沿 X 网格轴**未归约**（unreduced）”，表示该运算在某种意义上处于“未完成”的状态，要等最后求和之后才算完成。$\cdot_\text{LOCAL}$ 语法表示我们执行本地求和，但让结果保持未归约状态。

这可以看作是关于矩阵乘法与外积的如下结论：

$$A \cdot B = \sum_{i=1}^{P} \underbrace{A_{:,i} \otimes B_{i,:}}_{\in \mathbb{R}^{n \times m}}$$

其中 ⊗ 表示外积。因此，如果 **X** 轴上的第 **i** 个 TPU 持有 **A** 的第 **i** 列和 **B** 的第 **i** 行，我们就可以做一次本地矩阵乘法得到 $$A_{:,i} \otimes B_{i,:} \in \mathbb{R}_{n\times m}$$。这个矩阵的每个元素，都是 **A • B** 在该位置的求和式中的第 **i** 项。我们仍然需要在 **P** 上求和（而 **P** 被分片到了网格轴 **X** 上），才能得到完整的 **A • B**。如果我们把 **A** 和 **B** 按块（即分片）来写，再对结果的每个分片求和，道理也是一样的。

为了解决这个问题，我们可以通过在 **X** 轴上做一次完整的 **AllReduce** 来完成这个求和：

$$\begin{align*}
A[I, J_X] \cdot_\text{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{AllReduce}_X C[I, K] \{ U_X \} \rightarrow &\ C[I, K]
\end{align*}$$

AllReduce 消除部分和，使该轴上的*每个*设备都持有相同的、完全求和后的值。AllReduce 是本节要讨论的几种关键通信中的第二种，第一种是 AllGather，其余的是 ReduceScatter 和 AllToAll。AllReduce 接收一个带有未归约（部分求和）轴的数组，通过沿未归约的轴传递这些分片并累加结果来完成求和。它的签名是

$$\textbf{AllReduce}_Y A[I_X, J] \{U_Y\} \rightarrow A[I_X, J]$$

也就是说，它只是移除 $\\{U_Y\\}$ 后缀，其他方面保持结果不变。

**AllReduce 的开销有多大？** 理解 AllReduce 执行方式的一种心智模型是：每个设备把自己的分片发送给邻居，并把收到的所有分片加起来。显然，这比 AllGather 更昂贵，因为每个“分片”的形状都与完整数组相同。一般来说，**AllReduce 的开销是 AllGather 的两倍。** 理解这一点的一种方式是注意到 **AllReduce** 可以表示为另外两个原语的组合：一个 **ReduceScatter** 和一个 **AllGather**。与 AllReduce 一样，ReduceScatter 会消除数组上的部分和，但输出结果会沿某个给定维度“分散”（scatter），即被划分开。而 AllGather 会收集所有这些片段，把该逻辑轴沿对应的物理轴“取消划分/取消分片/复制”。

$$\begin{align*}
\textbf{ReduceScatter}_{Y,J} : A[I_X,J] \{U_Y\} \rightarrow &\ A[I_X, J_Y] \\
\textbf{AllGather}_Y : A[I_X, J_Y] \rightarrow &\ A[I_X, J]
\end{align*}$$

**那 ReduceScatter 呢？** 正如 AllGather 会重新组装一个分片的数组（移除一个下标），ReduceScatter 会对一个未归约/部分求和的数组求和，然后把另一个逻辑轴沿同一个网格轴分散（分片）：$X[F]\\{U_Y\\} \to X[F_Y]$。下面的动画展示了具体做法：注意它与 AllGather 非常相似，只是不再保留每个分片，而是把它们加在一起。因此，如果不计执行归约运算所花的时间，它的延迟与 AllGather 大致相同。

{% include figure.liquid path="assets/img/reduce-scatter.gif" class="img-fluid" %}

每一跳的通信时间就是每个分片的字节数 $V / Y$ 除以带宽 $W_\text{ici}$，与 AllGather 的情况相同，因此我们有

$$T_{\text{每次 AllGather 或 ReduceScatter 的通信量}} = \frac{V}{W_\text{ici}}$$

$$T_{\text{每次 AllReduce 的通信量}} = 2 \cdot \frac{V}{W_\text{ici}}$$

其中 $$W_\text{ici}$$ 是双向带宽，前提是我们有一个完整的环可以用来做归约。

### 情形 4：两个乘数都有一个非收缩维度沿同一个轴分片 {#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis}

对一个张量分片时，每个网格维度最多只能出现一次。套用上述规则有时会导致违反这条规则的情况，例如：

$$A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X]$$

这是无效的，因为沿维度 **X** 的某个分片（比如第 **i** 个）将持有 **C** 的第 **(i, i)** 个分片，也就是一个对角块。这样一来，所有分片中的信息加起来也只能恢复结果的对角块，因此我们不能允许这种分片方式。

解决办法是对某些维度做 AllGather。这里我们有两种选择：

$$\begin{align*}
\textbf{AllGather}_X A[I_X, J] \rightarrow &\ A[I, J] \\
A[I, J] \cdot B[J, K_X] \rightarrow &\ C[I, K_X]
\end{align*}$$

或者

$$\begin{align*}
\textbf{AllGather}_X B[J, K_X] \rightarrow &\ B[J, K] \\
A[I_X, J] \cdot B[J, K] \rightarrow &\ C[I_X, K]
\end{align*}$$

无论哪种情况，结果的形状中都只会出现一次 **X**。具体选哪一种，取决于后续运算需要什么样的分片方式。

## 深入理解 TPU 通信原语 {#a-deeper-dive-into-tpu-communication-primitives}

前面 4 种情形引入了几种用于执行分片矩阵乘法的“核心通信原语”：

1. **AllGather：**从分片中移除一个下标，把各个分片收集起来。
2. **ReduceScatter：**通过在某个轴上对分片求和来移除数组的“未归约”后缀，最终使数组沿第二个轴分片。
3. **AllReduce：**移除“未归约”后缀，让数组沿该轴不再分片。

还有一个核心通信原语需要介绍，它出现在混合专家（MoE）模型和其他一些计算中：**AllToAll**。

### 最后一个通信原语：AllToAll {#our-final-communication-primitive-the-alltoall}

最后一个基本集合通信是 **AllToAll**，它在考虑分片矩阵乘法时不会自然出现，但在实践中却经常遇到；更准确地说，它是*分片转置*或重新分片（resharding）操作的特例，例如：

$$\textbf{AllToAll}_{X, J} A[I_X, J] \rightarrow A[I, J_X]$$

AllToAll 通常用于在分片计算中布局方案互不兼容的不同区域之间重新排列分片布局。在考虑分片的混合专家模型时，它会很自然地出现。*你可以把 AllToAll 理解为把一个下标从一个轴移到另一个轴*。由于 AllToAll 不需要把每个分片的全部数据复制到环上的所有设备，它实际上比 AllGather *更便宜*（只需其 ¼ 的开销）<d-footnote>对于大小为偶数的双向环，每个设备向右发送 $(N/2 + (N/2-1) + … + 1)$ 个数据块，向左发送 $((N/2-1) + … + 1)$ 个数据块，合计 $= 0.5 \cdot (N / 2) \cdot (N/2 + 1) + 0.5 \cdot (N / 2) \cdot (N/2 - 1) = N^2/4$。每个数据块（也就是分片的分片）的大小为 $\text{字节} / N^2$，因此每个设备的开销为 $(\text{字节} / N^2) \cdot N^2 / 4 = \text{字节} / 4$。由于总带宽随设备数量线性增长，这一（每设备的）结果可以推广到所有设备上。</d-footnote>。

{% include figure.liquid path="assets/img/all-to-all.gif" class="img-fluid" %}

如果推广到 N 维 AllToAll，在 AxBxC 网格上，对于总字节数为 $V$（所有设备加总）的数组，总开销为

$$T_\text{每次 AllToAll 的通信量} = \frac{V \cdot \max(A, B, C, ...)}{4 \cdot N \cdot W_\text{ici}}$$

其中与往常一样，$W_\text{ici}$ 是双向 ICI 带宽，$N = A \cdot B \cdot C \cdot \ldots$ 是设备总数。等价地，用每个设备的字节数 $V / N$ 来表示，开销为 $(V / N) \cdot \max(A, B, C, ...) / (4 \cdot W_\text{ici})$。对于一维网格，它简化为 $V / (4 \cdot W_\text{ici})$，即 AllGather 开销的 1 / 4。在二维情况下，开销实际上会随最小轴的大小而按比例下降。

*题外话：如果你想要这一结论的粗略推导，可以从一维环面 $\mathbb{Z} / N\mathbb{Z}$ 开始。如果随机选取一个源节点和一个目标节点，它们之间平均相距 N / 4 跳，由此得到开销为 $(V \cdot N) / (4 * N)$。现在考虑 N 维环面，每个轴基本上是相互独立的。每个节点持有 $1 / N$ 字节，平均需要把数据传送 $\max(A, B, C, …) / 4$ 跳。你也可以从二分带宽（bisection bandwidth）推导出这一结论：在 AllToAll 中，网格的每一半都要把它一半的数据（$V / 4$ 字节）发送给另一半。最窄的二分切割垂直于最长的轴，穿过 $2 \cdot N / \max(A, B, …)$ 条链路（两个切割面，计入环绕连接），单方向带宽为 $N \cdot W_\text{ici} / \max(A, B, …)$。两者相除就得到上面的公式。*

### 关于 ReduceScatter 的更多内容 {#more-about-the-reducescatter}

ReduceScatter 是一个比初看起来更基本的操作，因为它实际上是 AllGather 的导数，反之亦然。也就是说，如果在前向传播中我们有：

$$\textbf{AllGather}_X A[I_X] \rightarrow A[I]$$

那么我们对反向模式导数 **A'**（它们在每个分片上通常各不相同）做 ReduceScatter，就得到分片的 **A'**：

$$\textbf{ReduceScatter}_X A'[I] \{ U_X \} \rightarrow A'[I_X]$$

同样，前向传播中的 $$\text{ReduceScatter}_X(A[I] \{U_X\}) \to A[I_X]$$ 意味着反向传播中的 $$\text{AllGather}_{X}(A'[I_X]) \to A'[I]$$。

{% details 点击此处查看 AllGather 与 ReduceScatter 如何互为导数的细节。 %}

这源于这样一个事实：作为线性算子，广播（broadcast）和归约（reduce）互为转置，而 AllGather 和 ReduceScatter 分别是广播和归约的外积（也称为 [Kronecker 积](https://en.wikipedia.org/wiki/Kronecker_product)）。具体来说，如果有一个向量 $x \in \mathbb{R}^n$、任意数量的设备 $p \in \mathbb{N}$，并令 $u = (1, \ldots, 1) \in \mathbb{R}^p$，我们可以按如下方式定义广播和归约，这应该与你对它们的直观理解一致：

$$
\begin{align*}
\text{广播} &: \mathbb{R}^n \rightarrow \mathbb{R}^{p n} \\
\text{广播} &= u \otimes \mathbf{I}_n \\
\text{归约} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^n \\
\text{归约} &= u^T \otimes \mathbf{I}_n
\end{align*}
$$

我们通过一个例子来看看它是什么样子，其中 $n = 1$，$p = 2$。如果 $x = (7)$，我们有 $$\text{广播}(x) = \left(\begin{pmatrix} 1 \\ 1 \end{pmatrix} \otimes \begin{pmatrix} 1 \end{pmatrix}\right) x = \begin{pmatrix} 1 \\ 1 \end{pmatrix} x = \begin{pmatrix}  7\\  7  \end{pmatrix} \in \mathbb{R}^{p n}$$。这与我们的预期一致：把 $\mathbb{R}^n$ 中的向量广播到 $\mathbb{R}^{pn}$。现在令 $y = (8, 9)$，我们有 $$\text{归约}(y) = \left(\begin{pmatrix} 1 & 1 \end{pmatrix} \otimes \begin{pmatrix} 1\end{pmatrix}\right) y = \begin{pmatrix} 1 & 1  \end{pmatrix} \begin{pmatrix}  8 \\ 9  \end{pmatrix} = \begin{pmatrix}   17    \end{pmatrix}$$。这同样符合预期：把 $\mathbb{R}^{p n}$ 中的向量归约为 $\mathbb{R}^{n}$ 中的向量。由于对任意两个矩阵 $A$ 和 $B$ 都有 $(A \otimes B)^T = A^T \otimes B^T$，我们可以看出 $\text{归约} = \text{广播}^T$。AllGather 和 ReduceScatter 则可以表示为如下外积：

$$
\begin{align*}
\text{AllGather} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^{p^2 n} \\
\text{AllGather} &= \text{广播} \otimes \mathbf{I}_p \\
\text{ReduceScatter} &= \mathbb{R}^{p^2 n} \rightarrow \mathbb{R}^{p n} \\
\text{ReduceScatter} &= \text{归约} \otimes \mathbf{I}_p
\end{align*}
$$

这里我们把 $\mathbb{R}^{p^2 n}$ 看作 $\mathbb{R}^{p \times p n}$，即 $p$ 个设备中的每一个都对应一个 $\mathbb{R}^{p n}$ 向量。我们建议你动手试几个小例子，比如 $n = 2$、$p = 3$，看看这些算子写成矩阵是什么样子。利用同样的转置性质，我们再次得到 $\text{AllGather}^T = \text{ReduceScatter}$，当然也有 $\text{ReduceScatter}^T = \text{AllGather}$。这种转置会在反向传播中出现：如果对于某个线性算子 $A$（例如 AllGather 或 ReduceScatter）有 $y = Ax$，那么在反向传播时我们会得到损失对 $y$ 的导数 $\frac{\partial L}{\partial y}$，并通过 $\frac{\partial L}{\partial x} = A^T \frac{\partial L}{\partial y}$ 得到 $\frac{\partial L}{\partial x}$。这就说明了 AllGather 的导数为什么是 ReduceScatter，反之亦然。

{% enddetails %}

把 AllReduce 拆成 AllGather 和 ReduceScatter 还有一个便利之处：我们可以把最后的 AllGather 推迟到之后某个时刻再做。很多时候，我们并不想付出代价，把完整的矩阵乘积重新组装并复制到所有设备上。相反，即使是在两个乘数都有分片收缩维度的这种情况下，我们也希望保持分片状态：

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K_X]$$

在这种情况下，我们也可以用 ReduceScatter 代替 AllReduce，然后视需要在之后某个时刻再做 AllGather，即

$$\begin{align*}
A[I, J_X] \cdot_{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{ReduceScatter}_{X,K} C[I, K] \{ U_X \} \rightarrow &\ C[I, K_X]
\end{align*}$$

注意，ReduceScatter 会*引入*一个分片维度，因此在这里天然可以自由选择沿命名维度 **I** 或 **K** 分片。使用 ReduceScatter 时，我们通常需要选择在*哪个*命名维度上引入新的分片（尽管这一选择通常由更大的建模上下文所决定）。这就是为什么我们使用 **ReduceScatter<sub>X,K</sub>** 这样的语法来指定要分片的轴。

### 如何让矩阵乘法的通信与计算重叠 {#how-to-overlap-matmul-communication-with-compute}

正如我们在[第 1 部分](../roofline)中讨论的，我们通常假设只要通信足够快，就总能让通信与某些有用的计算重叠。本节中的集合通信通常可以与矩阵乘法计算本身重叠，但做到这一点并不简单。我们使用的算法叫作**集合矩阵乘法（collective matmul）**，最早由 [Wang et al.](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959) 提出。下面是一个简化的动画，展示了如何实现这种重叠：

{% include figure.liquid path="assets/img/ag_matmul.gif" caption="<b>图：</b>演示如何让单个分片矩阵-向量乘积与随之而来的 AllReduce（上文的情形 3）重叠的动画。一次完整的矩阵乘法由多个矩阵-向量乘积组成。" %}

简单来说，我们可以在对矩阵的一个块做矩阵乘法的同时，开始对之前的块进行环形归约。在某些情况下，我们还可以在批次维度或矩阵输入维度上进行分块（tile）。我们会在[第 10 部分](../jax-stuff)中给出一个简单的 JAX 实现，[Mosaic 文档](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)也给出了一个很好的 GPU 示例。我们鼓励你在某个时候自己实现一个版本。

## 我们学到了什么？ {#what-have-we-learned}

* 数组的分片由两部分来指定：一个 **Mesh**，它为 TPU 网格（mesh）的物理硬件轴命名；以及一个 **Sharding**，它把网格轴名称分配给数组的逻辑轴。
  * 例如，**A**[I<sub>XY</sub>, J] 描述了一个抽象数组 **A**，它的第一个维度沿 X 和 Y 两个网格轴分片。再配合 Mesh(mesh_shape=(4, 8), axis_names=('X', 'Y')) 或其简写 Mesh({'X': 4, 'Y': 8})，就可以知道该数组沿第一个维度被切成了 32 份。

* **对已分片数组做运算，与对未分片数组做运算完全一样，除非你沿一个已分片的轴做收缩（contraction）**。这种情况下就必须引入一些通信。我们考虑四种情况：

  1. *两个数组都没有沿收缩维度分片*：不需要任何通信。
  2. *一个数组沿收缩维度分片*（或者两个数组的收缩维度沿不同的轴分片）：在执行运算之前，先对其中一个输入做 AllGather。
  3. *两个数组沿收缩维度以相同方式分片：* 先在本地对分片做乘法，然后执行 AllReduce 或 ReduceScatter。
  4. *两个数组在非收缩维度上沿同一个网格轴分片：* 先对其中一个输入做 AllGather。

* TPU 大致使用 **4 种核心通信原语**：
  1. AllGather：$[A_X, B] \to [A, B]$
  2. ReduceScatter：$[A, B] \\{U_X\\} \to [A_X, B]$
  3. AllToAll：$[A, B_X] \to [A_X, B]$
  4. AllReduce：$[A_X, B]\\{U_Y\\} \to [A_X, B]$（严格来说它不是原语，因为它是 ReduceScatter + AllGather 的组合）

{% include figure.liquid path="assets/img/all-collectives.png" class="img-fluid" %}

* 这些操作各自的开销和延迟**与轴的大小无关（只要它们是带宽受限的）**，只取决于输入数组的大小和链路带宽。对于单向的 AllGather/ReduceScatter：

$$T_{\text{每次 AllGather 或 ReduceScatter 的通信量}} = \frac{\text{数据量}}{\text{带宽}} \cdot \frac{\text{轴} - 1}{\text{轴}}
\longrightarrow \frac{\text{数据量}}{\text{带宽（双向）}}$$

* AllReduce 由一次 ReduceScatter 加一次 AllGather 组成，因此开销是上面的 2 倍。AllToAll 只需把分片沿环传递一部分路程，因此开销是 AllGather 的 ¼。总结如下：

| 操作         | 描述                                                                                                        | 语法                           | 运行时间                                          |
| :---------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------- | :----------------------------------------------- |
| **AllGather**     | 沿某个轴收集已分片数组的所有分片，去掉一个下标。                                     | $[A_X, B] \to [A, B]$            | 字节数 / (双向 ICI 带宽 * 轴数) |
| **ReduceScatter** | 沿某个轴对部分求和的数组求和，并沿另一个轴对其分片（增加一个下标）。                 | $[A, B] \\{U_X\\} \to [A_X, B]$  | 与 AllGather 相同                                |
| **AllReduce**     | 沿某个轴对部分求和的数组求和。去掉一个 { U<sub>x</sub> }。由一次 AllGather 和一次 ReduceScatter 组合而成。 | $[A_X, B]\\{U_Y\\} \to [A_X, B]$ | 2 * AllGather                                    |
| **AllToAll**      | 收集（复制）某个轴，并沿同一个轴对另一个维度分片。                                 | $[A, B_X] \to [A_X, B]$          | 双向环上为 AllGather / 4           |

## 一些练习题 {#some-problems-to-work}

*下面是一些基于本节内容的有启发性的题目。目前我们不会给出所有答案，但会尽可能补充更多答案。*

**问题 1 [复制分片]**：一个数组按 $A[I_X, J, K, \ldots]$ 分片（即只沿 $X$ 分片），网格为 `Mesh({'X': 4, 'Y': 8, 'Z': 2})`。$A$ 在所有芯片上占用的总字节数与该数组单份副本大小之比是多少？

{% details 点击此处查看答案。 %}

我们的数组只沿大小为 4 的 X 分片，因此实际上每个分片的大小为 $[I / 4, J, K, \ldots] = \text{sizeof}(A) / 4$。由于数组在 Y 和 Z 上是复制的，总大小为 $Y \cdot Z \cdot \text{sizeof}(A)$，所以总大小与单芯片大小之比为 $Y \cdot Z \cdot \text{sizeof}(A) / \text{sizeof}(A) = 16$。

{% enddetails %}

**问题 2 [AllGather 延迟]**：在网格为 `Mesh({'X': 4, 'Y': 4, 'Z': 4})` 的 TPU v4p 4x4x4 切片上，如果 $B=1024$、$D=4096$ 且使用 bfloat16，$\text{AllGather}_X([B_X, D_Y])$ 需要多长时间？$$\text{AllGather}_{XY}([B_X, D_Y])$$ 呢？$$\text{AllReduce}_Z([B_X, D_Y] \{U_Z \})$$ 呢？

{% details 点击此处查看答案。 %}

因为我们有一个完整的 `4x4x4` 立方体，所以所有轴上都有环绕连接（wraparound），可用的双向带宽为 9e10。

1. 因为我们只在一个轴上做收集，而另一个轴是分片的，所以实际上是在 1 个轴上收集 $2BD / Y$ 字节。*如果只考虑沿 Y 轴的单个分片，沿 X 的 AllGather 看起来就像一个字节数为 1 / Y 的未分片 AllGather。* 由于 TPU v4p 的 ICI 双向带宽为 9e10 字节/秒，这将花费 $2BD / (\text{9e10} \cdot Y) = 2 \cdot 1024 \cdot 4096 / (\text{9e10} \cdot 4) = 23 \mu s$。

2. 带宽是之前的两倍，但我们要对整个数组做 AllGather，所以 `T = 2BD / (2 * W) = 2*1024*4096 / (2 * 9e10) = 46us`。这远高于 4us 的延迟下界（每跳 1us），所以没有问题。

3. AllReduce 的开销是 AllGather 的两倍。每个分片的大小为 $2BD / (X * Y)$，因此开销约为 $4BD / (X * Y * W)$，大约是 `4 * 1024 * 4096 / (16 * 9e10) = 11.6us`。

*趣闻：* 第 (1) 和第 (2) 部分其实并不是最优的，因为数组还沿未使用的 Z 轴复制，我们可以利用那些空闲的链路：可以先免费地重新分片 $[B_X, D_Y] \to [B_{XZ}, D_Y]$（每个设备只需丢弃自己分片的一部分），然后执行 $$\text{AllGather}_{XZ}$$（或 $$\text{AllGather}_{XYZ}$$），在更多的轴上做收集的同时达到相同的最终状态。这样第 (1) 部分可降到 11.5us，第 (2) 部分可降到 31us——实践中，只要一开始就在更多轴上分片就能得到这个效果，这也是要尽可能细粒度地对数组分片的原因之一。

{% enddetails %}

**问题 3 [延迟受限的 AllGather]**：假设我们要执行 $\text{AllGather}_X([B_X])$，但 $B$ 非常小（比如 128）。在网格为 `Mesh({'X': 4, 'Y': 4, 'Z': 4})` 的 TPU v4p 4x4x4 切片上，使用 bfloat16，这需要多长时间？*提示：你很可能是延迟受限的。*

{% details 点击此处查看答案。 %}

我们的数组在 bfloat16 下总共只占 256 字节，每个设备只有 64 字节。由于在 TPU v4p 上该轴大小为 4，我们有环绕连接，因此可以双向发送数组。单向带宽为 `4.5e10`，每一跳大约需要 `64 / 4.5e10 ~ 0`，所以我们肯定是延迟受限的。数一下跳数，只需 2 跳就能完成整个收集，因此大约 2us 是一个不错的估计。

{% enddetails %}

**问题 4 [矩阵乘法策略]**：为了执行 $X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$，本节告诉你要先执行 $\text{AllGather}_X(Y[D_X, F])$，再将完全复制的矩阵相乘（情况 2，*策略 1*）。另一种做法是，像 $X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F] \\{U_X\\}$ 这样对本地分片做乘法（情况 3，*策略 2*），然后执行 $\text{AllReduce}_X(Z[B, F] \\{ U_X\\})$。这两种策略各自需要多少 FLOPs 和通信？哪个更好，为什么？

{% details 点击此处查看答案。 %}

先从基线（*策略 1*）开始。如前所示，AllGather 的开销为 $2DF / W_\text{ici}$。拿到完全复制的数组后，总计算时间为 $2BDF / C$（其中 $C$ 是加速器的 FLOPs/s，因为每个 TPU 执行的 FLOPs 相同）。于是有

$$T_\text{总计（策略 1）} = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_\text{ici}}\right)$$

相比之下，新策略（策略 2）对 $2BF$ 字节做 AllReduce，开销为 $4BF / W_\text{ici}$，但 FLOPs 只有原来的 $1 / X$（因为计算是分片的）。这意味着我们执行 $2\cdot B\cdot D\cdot F / X$ 次 FLOPs，而随后的 AllReduce 在 bfloat16 下通信 $$2 \cdot 2 \cdot B \cdot F$$ 字节。因此，*策略 2*（不做 AllGather，只在之后做一次 AllReduce）的总时间大约为

$$T_\text{total} = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_\text{ici}}\right)$$

问题在于：*哪一个更大？* 当 $D / (X \cdot C) > 2 / W_\text{ici}$，即 $D / 2X > C / W_\text{ici} \approx 2550 \rightarrow X < D / (2 * 2550)$ 时，策略 (2) 是计算受限的。我们可以合理地预期 $D \approx 8k$，这意味着大约需要 $X < 2$，这不太可能——因此使用策略 2 时我们基本上总是通信受限的。对于基线（策略 1），当 $$B < C / W_\text{ici} = 2550$$ 时我们是通信受限的，这经常成立，但并非总是如此。

所以如果 $B < 2550$，两种情况下我们都是通信受限的，此时有

$$T_\text{策略 2 的通信} < T_\text{策略 1 的通信} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2DF}{W_\text{ici}}$$

当 $D > 2B$ 时上式成立，其中 $2B < 5100$。这通常成立，所以当批次较小时，策略 2 有时会更好。当批次较大（$B > 2550$）时，有

$$T_\text{策略 2 的通信} < T_\text{策略 1 的计算} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2BDF}{C}$$

当 $2 / W_\text{ici} < D / C$，即 $D > 2 * 2550 = 5100$ 时上式成立，这对大模型通常成立。所以除非 $D$ 很小，否则这种替代策略对大模型通常更好。

*那为什么我们不总是这样做？* 嗯，实践中我们有时确实会这样做，但一个矩阵乘法的某个输入的收缩维度沿某个轴分片、而另一个输入却没有沿该轴分片，这种情况通常很少见。例如，如果我们在做 FSDP（在[第 5 部分](../training)中解释），我们会沿数据维度对参数分片，但激活值*也会沿数据维度分片*。所以从这个意义上说，这种情况并不常出现。

{% enddetails %}

**问题 5 [最小延迟]**：假设我想在 TPU v4p 4x4x4 上以尽可能低的延迟执行矩阵乘法 $A[I, J] \cdot_J B[J, K] \to C[I, K]$。假设输入可以任意分片，但结果应当是完全复制的。输入应该如何分片？总的 FLOPs 和通信时间是多少？

{% details 点击此处查看（部分）答案。 %}

这里我们不提供完整答案，但会先描述四种最可能的方案：

1. $A[I_{XYZ}, J] \cdot B[J, K]$ + 最后做 AG
2. $A[I, J] \cdot B[J, K_{XYZ}]$ + 最后做 AG
3. $A[I, J_{XYZ}] \cdot B[J_{XYZ}, K]$ + 最后做 AR
4. $A[I, J] \cdot B[J, K]$（完全复制）

我们也可以考虑把不同的轴沿不同的网格轴分片，但这不太可能改变最终开销。除 (4) 之外，每个 TPU 的总 FLOPs 都相同，但各自的通信量不同。接下来只需计算每种方案的通信开销，看哪个最低即可。简而言之（TLDR），(1) 和 (2) 一样好。

{% enddetails %}

**问题 6：** 假设我们要在 TPU v5e 4x4 上执行 $A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$。需要执行什么通信？通信时间和计算时间各是多少？

* $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$ 呢？这是训练中最标准的设置，其中结合了数据、张量和 ZeRO 分片。
* $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$ 呢？这是推理中的标准设置，我们只做纯张量并行（+数据并行）。

**问题 7：** 一个典型的 Transformer 块有两个矩阵 $W_\text{in}[D, F]$ 和 $W_\text{out}[F, D]$，其中 $F \gg D$。假设批大小为 B，那么完整的块就是 $In[B, D] \cdot W_\text{in}[D, F] \cdot W_\text{out}[F, D]$。取 $D=8192$、$F=32768$、$B=128$，并假设所有数据都是 bfloat16。假设我们运行在 TPU v5e 2x2 切片上，但不妨假定每个 TPU 只有 300MB 空闲内存。In、$W_\text{in}$、$W_\text{out}$ 和 Out 应该如何分片，才能在不超出内存限制的同时使总时间最小？通信和 FLOPs 各花费多少时间？*提示：最终输出不需要完全复制，但它的分片方式应与输入相同，这样这个“层”才能重复堆叠。*

{% details 点击此处查看（部分）答案。 %}

首先考虑内存。两个大矩阵各占用 `2 * 8192 * 32768 = 536MB`。激活值 `In` 的大小为 `2 * 128 * 8192 = 2MB`（小到无需担心）。由于每个设备只有 300MB 空闲内存，我们显然需要对矩阵乘法做分片。

1. $In[B_X, D] * W_\text{in}[D_{XY}, F] * W_\text{out}[F, D_{XY}] \rightarrow Out[B_X, D]$（这通常被称为 FSDP）
2. $In[B, D_{XY}] * W_\text{in}[D, F_{XY}] * W_\text{out}[F_{XY}, D] \rightarrow Out[B, D_{XY}]$（这被称为张量并行）

第一种相当糟糕，因为我们需要先对大权重或激活值做 AllGather。第二种需要在开头做一次 AllGather，在结尾做一次 ReduceScatter（比 AllReduce 更便宜）。剩下的计算就留作练习吧。

{% enddetails %}

**问题 8 [挑战题]**：以上面的简短代码片段为模板，分配一个已分片的数组，并使用 pmap 或 shard_map 对 4 种主要通信原语（AllGather、AllReduce、ReduceScatter 和 AllToAll）分别做基准测试。你会用到 `jax.lax.all_gather`、`jax.lax.psum`、`jax.lax.psum_scatter` 和 `jax.lax.all_to_all`。你理解这些函数的语义吗？它们各需要多长时间？

**问题 9 [已分片矩阵乘法的另一种策略？]**：我们在[上文](#case-2-one-multiplicand-has-a-sharded-contracting-dimension)中声称，当矩阵乘法只有一个输入沿其收缩维度分片时，应该对这个已分片矩阵做 AllGather，然后在本地执行收缩。你可能会想到的另一种策略是：先执行已分片的矩阵乘法，然后对结果做 AllReduce（就好像两个输入都沿收缩维度分片一样），即通过以下步骤实现 $A[I, J_X] *_J B[J, K] \to C[I, K]$：

1. $C[I, K] \\{ U_X \\} = A[I, J_X] \cdot B[J_X, K]$
2. $C[I, K] = \text{AllReduce}(C[I, K] \\{ U_X\\})$

回答以下问题：

1. 针对矩阵 $A[N, M]$ 和 $B[M, K]$ 显式写出这个算法，用下标准确说明每个设备上执行了什么计算。假设 $A$ 以 $A[I, J_X]$ 的方式分片在 ND 个设备上，并且你希望输出在所有设备上都是复制的。
2. 现在假设你可以接受最终结果不在每个设备上复制，而是分片的（沿 N 或 K 维度）。上面的算法会如何变化？
3. 只看上述策略（第 2 部分的，而不是第 1 部分的）的通信开销，它与先对 A 做 AllGather 再做矩阵乘法的算法相比如何？

{% details 点击此处查看答案。 %}


1. 首先计算外积，并将结果存储在 $$O[N, K]: o_{kj} = \sum_i a_{ki} b_{ij}$$ 中。注意，由于我们在做外积，重复出现的下标并不是被收缩的那个。这里的求和范围是我们所用的特定设备上存储的那组 i 值。例如，如果收缩轴大小为 16，有 4 个设备，那么在设备 0 上，i 的取值范围是 {0, 1, 2, 3}；在设备 1 上是 {4, 5, 6, 7}；在设备 2 上是 {8, 9, 10, 11}；在设备 3 上是 {12, 13, 14, 15}。然后对每个设备上 $O[N, K]$ 的部分和做 AllReduce，得到完整的 $O[N, K]$。
2. 在第 2 步中，我们不必做 AllReduce，而是可以用更便宜的 ReduceScatter，沿任一轴均可：$[N, K] \\{ U_X \\} \to [N_X, K]$ 或 $[N, K] \\{ U_X \\} \to [N, K_X]$。
3. 正如上面正文所述，做 AllGather 的开销（在吞吐量受限时）与 ReduceScatter 相同；它只由我们处理的完整矩阵的大小决定。因此在“先 gather 再做矩阵乘法”的算法中，开销随 $NM$ 增长（因为我们对 $A$ 做 $\text{AllGather}$）；在“先做矩阵乘法再 reduce-scatter”的算法中，开销随 NK 增长（因为我们对 $O$ 做 reduce-scatter）。所以两种算法的通信开销之比为 `M/K`。

{% enddetails %}

**问题 10：玩转 AllToAll：** 在上面的表格中提到，执行 AllToAll 的时间只有执行 AllGather 或 ReduceScatter 的时间的 1/4（在吞吐量受限的情形下）。在这道题中，我们将看看这个 4 倍的因子是从哪里来的，并看看如果我们只有单向 ICI 链路而不是双向 ICI 链路，这个倍数会如何变化。

1. 我们先从单向的情况开始。设想我们有 *D* 个设备组成环形拓扑，想要对一个 N x N 矩阵 $A[I_X, J]$ 做 AllGather 或 ReduceScatter（为简单起见，假设 $D$ 整除 $N$）。描述这两种集合通信所涉及的通信，并计算在整个算法过程中经过**单条** ICI 链路传输的标量（浮点数或整数）总数。
2. 现在考虑 AllToAll，仍然是单向 ICI 的情况。这种情况下的算法与 all-gather 有何不同？计算该算法中经过单条 ICI 链路传输的标量数量。
3. 你应该会发现第 (a) 部分与第 (b) 部分答案之比是一个漂亮的数字。用简单的语言解释这个因子从何而来。
4. 现在加入双向通信。这对 all-gather 情况下所需的总时间有何影响？
5. 加入双向通信对 AllToAll 情况下所需的总时间有何影响？
6. 现在简单解释一下在双向环上 AllGather 时间与 AllToAll 时间之比。

{% details 点击此处查看答案。 %}

(1) **解答：** 过程很简单：在算法的每一步中，每个设备都会把矩阵的一个单分片“条带”（共 $$\frac{N}{D} \times N$$ 个元素）发送给它最近的邻居。这会发生 $$D-1$$ 次，因为每个分片都需要被传送到除其初始所在设备以外的所有设备。所以总共每个设备传输 $$\frac{N^2(D-1)}{D}$$ 个标量，即经过单条 ICI 链路的标量数。

**答案：** $$N^2 (1-\frac{1}{D})$$，当 $$D >> 1$$ 时就是 $$N^2$$。

(2) **解答：** 从通信的角度看，AllToAll 与 AllGather 的关键区别在于：在 AllToAll 中，位于某个设备上的分片并不需要整个传送给其他每个设备。设想存储在某个设备（称之为设备 0）上的分片为 $$[A, B, C, D]$$（这里 A、B、C、D 是矩阵，为便于说明，我们设想一个有 4 个设备的环）。那么矩阵 $$A$$ 不需要传送到任何地方，矩阵 $$B$$ 最终需要到达设备 1；矩阵 $$C$$ 到达设备 2；矩阵 $$D$$ 到达设备 3。所以在算法的第一步，我们把 $$B$$、$$C$$ 和 $$D$$ 发送到设备 1；下一步，设备 1 把 $$C$$ 和 $$D$$ 继续发送到设备 2；最后一步，设备 2 只把 $$D$$ 发送到设备 3。这种情况下传输的参数总数为 $$(\text{A/B/C/D 的大小}) * (3 + 2 + 1)$$。（现在推广到一般情况）A/B/C/D 的大小为 $$\frac{N^2}{D^2}$$，同样在一般情况下 $$(3 + 2 + 1)$$ 这一项变为 $$((D-1) + (D-2) + … + 1)$$，即 $$\frac{(D)(D-1)}{2}$$。所以经过单条 ICI 链路传输的总字节数为 $$\frac{N^2(D-1)}{D \times 2}$$。

**答案：** $$\frac{N^2}{2}(1-\frac{1}{D})$$，当 $$D >> 1$$ 时就是 $$\frac{N^2}{2}$$。

(3) **解答：** 这个因子就是 $$\frac{1}{2}$$，即在单向环形拓扑上，AllToAll 的开销是 all-gather/ReduceScatter 的一半。回顾上面的推导，这最终来自这样一个事实：在 all-gather 的情况下，我们把同样大小的块各传输 $$(D-1)$$ 次，即我们在计算求和 $$ \text{极小块大小} * (D + D + D + … + D)$$；而在 AllToAll 的情况下，我们计算的是求和 $$\text{极小块大小} * (D + D-1 + D-2 + … + 1)$$。因此这个 2 倍的因子本质上来自 $$1 + 2 + \ldots + n = n(n+1)/2$$ 这一事实。

(4) **解答**：任意一条链路需要承载的标量总数现在减少为原来的一半，因为在双向环上，每个“分片条带”可以同时向两个方向发送。

(5) **解答**：这种情况下，与单向情况相比我们获得了 4 倍的提升。考虑单个分片条带（比如起始于设备 0 的那个）中每个大小为 (N2/D2) 的块的去向，就最容易理解这一点。我们不再像单向情况那样把一个块发送 D-1 的距离、另一个块发送 D - 2 的距离，依此类推直到 1，而是把条带划分成向右或向左移动的块，每块最多移动 floor(D/2) 的距离。于是相应的求和变为 $$D/2 + D/2 - 1 + D/2 - 2 + … = D/2 \cdot (D/2+1)/2$$，在 $$D$$ 很大的极限下即为 $$D^2/8$$。将其与单向情况下的 $$D^2/2$$ 相比，可以看到我们获得了 4 倍的提升。

(6) **解答：** 在单向环上，我们看到 AllToAll 的时间已经是 all-gather 时间的一半；这是因为我们不需要把完整的条带发送给每一个设备。然后，当我们加入双向通信后，我们看到 AllToAll 获得了 4 倍的提升，而 all-gather 只获得了 2 倍的提升。把这些比值结合起来，就得到了我们所求的 4 倍因子。

{% enddetails %}

<h3 markdown=1 class="next-section">第 3 部分到此结束！第 4 部分（关于 Transformer 数学）请点击[这里](../transformers)！</h3>
