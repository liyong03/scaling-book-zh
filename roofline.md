---
layout: distill
title: "关于 Roofline 的一切"
# permalink: /main/
description: "当我们在硬件上运行算法时，会受到三方面的限制：计算机做数学运算的速度（OPs/second）、搬运数据可用的带宽（bytes/second），以及可用于存储数据的总内存（bytes）。这些 “Roofline” 约束让我们能够给出某次计算耗时的上界和下界。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 1

previous_section_url: ".."
previous_section_name: "第 0 部分：引言"

next_section_url: ../tpus
next_section_name: "第 2 部分：TPU"

bibliography: main.bib

giscus_comments: false
giscus_discussion: 3

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

  - name: 时间都花在哪儿了？
    id: where-does-the-time-go
  - subsections:
    - name: "Roofline 可视化"
      id: visualizing-rooflines
    - name: "矩阵乘法"
      id: matrix-multiplication
    - name: "网络通信的 Roofline"
      id: network-communication-rooflines
  - name: 几道练习题
    id: a-few-problems-to-work

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

## 时间都花在哪儿了？ {#where-does-the-time-go}

我们从一个极其简单的问题开始：*为什么一个算法耗时 50ms，而不是 50s 或 5ms*？模型内部究竟发生了什么，需要耗费大量时间？我们又应该预期它花多长时间？

**计算：** 深度学习模型本质上就是一堆矩阵乘法，每个矩阵乘法都由浮点乘法和加法“运算”（FLOPs）组成。加速器的速度决定了计算这些需要多长时间：

$$\begin{equation}
T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}}
\end{equation}$$

例如，一块 NVIDIA H100 每秒大约可以执行 9.89e14 次 bfloat16<d-footnote>bf16 是 <a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">bfloat16</a> 的缩写，这是一种在机器学习中常用的 16 位浮点格式。</d-footnote> FLOPs，而一块 TPU v6e 可以执行 9.1e14 FLOPs/s。<d-footnote>H100 和 B200 通常只能达到标称峰值 FLOPs 的 80-85% 左右，而 TPU 在正常使用中可以更接近 95%。</d-footnote> 这意味着在 H100 上执行 1e12 FLOPs（大约）需要 `1e12 / 9.89e14 = 1.01ms`，而在 TPU v6e 上需要 `1e12 / 9.1e14 = 1.1ms`。<d-footnote>注意，这些芯片的定价不同，这里的比较并没有按成本做归一化。</d-footnote>

**芯片内部的通信：** *在一个加速器内部*，张量需要在加速器内存（HBM）和计算核心之间传输。你会看到这条链路的带宽被称为 “HBM 带宽”。<d-footnote>NVIDIA 也把它称为 “内存带宽（memory bandwidth）”。</d-footnote> 在 H100 上，[它约为 3.35TB/s](https://www.nvidia.com/en-us/data-center/h100/)，在 TPU v6e 上[约为 1.6TB/s](https://cloud.google.com/tpu/docs/v6e)。

**芯片之间的通信：** 当我们把模型分布到*多个*加速器上时，张量经常需要在它们之间传输。我们的硬件上通常有几种可选方式（ICI、DCN 和 PCIe），各自的带宽不同。

无论通信发生在芯片内部还是芯片之间，我们都以 bytes/s 来衡量，并用下式估算总通信时间：

$$\begin{equation}
T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Network/Memory Bandwidth Bytes/s}}
\end{equation}$$

通常（但并非总是如此），单个芯片内的计算可以与芯片内及芯片间的通信相互重叠。这意味着**我们可以用计算时间和通信时间中的最大值作为训练和推理时间的下界**。我们也可以**用它们之和作为上界**。在实践中，我们以最大值为目标进行优化，因为这样代数更简单，而且通过重叠通信与计算，我们通常可以相当接近这个界。如果我们以最大值为目标进行优化，那么上下界之间最多相差 2 倍，因为 $T_\text{math} + T_\text{comms} \leq 2 * \max(T_\text{math}, T_\text{comms})$。在此基础上，我们还可以通过对“重叠区域”和各种开销建模来进一步提高准确性，而这可以借助对你的具体模型和目标系统做性能分析（profiling）来获得依据。

$$\begin{equation}
T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}$$

$$\begin{equation}
T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}$$

如果假设我们可以完美地重叠通信与计算，那么当 $T_\text{math} > T_\text{comms}$ 时，硬件就能被充分利用。我们称之为“计算受限”（compute-bound）。当 $T_\text{comms} > T_\text{math}$ 时，我们往往处于“通信受限”（communication-bound）状态<d-footnote>在本书中，我们会交替使用 “通信受限”（communication-bound、comms-bound）、“内存受限”（memory-bound）和 “带宽受限”（bandwidth-bound）这几种说法。</d-footnote>，加速器的 FLOPs/s 至少有一部分被浪费在等待数据传输上。判断一个操作是计算受限还是通信受限的一种方法，是看它的“**算术强度**”（arithmetic intensity）或“**运算强度**”（operational intensity）。

**定义：** 一个算法的算术强度，是它执行的总 FLOPs 与它需要通信的字节数（无论是芯片内部还是芯片之间）之比。

$$\begin{equation}
\text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}
\end{equation}$$

算术强度衡量的是一个操作的“每字节 FLOPs 数”。粗略地说，当算术强度较高时，$T_\text{math}$ 相对于 $T_\text{comms}$ 较大，我们通常能用上大部分可用的 FLOPs。反之，我们就会在通信上花费更多时间，浪费 FLOPs。发生这种转折的点就是硬件的“峰值算术强度”，即加速器峰值 FLOPs/s 与加速器带宽之比。

$$\begin{align*}
T_\text{math} > T_\text{comms} \Leftrightarrow \frac{\text{Computation FLOPs}} {\text{Accelerator FLOPs/s}} > \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \frac{\text{Computation FLOPs}}{\text{Communication Bytes}} > \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \text{Intensity}(\text{Computation}) > \text{Intensity}(\text{Accelerator}) & \\
\end{align*}$$

$\text{Intensity}(\text{Accelerator})$ 这个量就是加速器达到峰值 FLOPs/s 时的算术强度。**对于 TPU v5e 的 MXU，这个值约为 240 FLOPs/byte**，因为 TPU 每秒可以执行 `1.97e14` FLOPs，并且每秒可以从 HBM 加载 `8.2e11` 字节。<d-footnote>MXU 是 TPU 上的矩阵乘法单元。这里特意说明这一点，是因为 TPU 上还有其他加速单元，比如负责逐元素运算的 VPU，它们的峰值 FLOPs/s 不同。</d-footnote> 这意味着如果一个算法的算术强度低于 240 FLOPs/byte，它就会受限于字节加载，因此我们无法充分利用硬件。<d-footnote>这只在算法从 HBM 加载权重并在 MXU 上运行时才成立。我们将在下一节讨论，有时可以把参数存放在 VMEM 中，它的带宽要高得多。许多算法也在 VPU 上运行，而 VPU 的性能特征不同。</d-footnote> 我们来看一个这样的例子：

**<span style="color:#7ab5ff">示例（点积）</span>：** 要以 bfloat16 精度计算两个向量的点积 `x • y: bf16[N], bf16[N] → bf16[1]`，我们需要从内存中加载 $x$ 和 $y$，它们各有 $2 * N = 2N$ 字节；执行 $N$ 次乘法和 $N-1$ 次加法；再把 $2$ 字节写回 HBM
$$\begin{equation}
\text{Intensity}(\text{dot product}) = \frac{\text{Total FLOPs}}{\text{Total Bytes}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}
\end{equation}$$

当 $N\rightarrow\infty$ 时成立。所以点积的算术强度为 $\frac{1}{2}$，换句话说，点积每加载一个字节只执行 0.5 次浮点运算。这意味着我们的算术强度低于硬件的算术强度，因此会是通信受限的。<d-footnote>上面的 240 在这里其实不是正确的比较对象，因为正如你将在下一节看到的，点积是在 VPU 而不是 MXU 上执行的。TPU v5p 的 VPU 每个核心每秒大约可以执行 7e12 FLOPs，所以它的临界算术强度约为 3，这意味着这里我们仍然有些通信受限。无论如何，我们的算术强度很低而且是常数，这意味着在大多数硬件上都很难做到计算受限。</d-footnote>

### Roofline 可视化 {#visualizing-rooflines}

我们可以用 **Roofline 图**来直观展示内存与计算之间的权衡。它以算法的算术强度为横轴（x 轴），以该算法在我们的硬件上可达到的峰值 FLOPs/s（吞吐量）为纵轴（y 轴）。下面是一个双对数坐标图的例子：

{% include figure.liquid path="assets/img/roofline-improved.png" class="img-fluid" caption="<b>图：</b>一个 Roofline 图示例，展示了两个算术强度不同的算法（Algo 1 和 Algo 2），以及它们在不同带宽（BW1 和 BW2）下对应的理论峰值吞吐量。在红色区域中，算法在两种带宽下都是带宽受限的，浪费了硬件峰值 FLOPs/s 的一部分。黄色区域只在较低带宽（BW1）下是带宽受限的。绿色区域在所有带宽下都是计算受限的。在这里，我们已经用满了加速器的峰值 FLOPs/s，提高带宽或改善算术强度都不会带来任何收益。" %}

在上图中，随着算术强度增加（从左向右移动），算法的性能（以 FLOPs/s 计）起初呈线性增长，直到达到硬件的临界算术强度——对 TPU v5e 而言是 240。任何算术强度更低的算法都会是带宽（BW）受限的，受峰值内存带宽的限制（红色所示）。任何位于其右侧的算法都能充分利用我们的 FLOPs（绿色所示）。这里，Algo 1 是通信受限的，只用到了硬件总 FLOPs/s 的一部分；Algo 2 则是计算受限的。一般来说，我们可以通过提高算法的算术强度，或者增加可用的内存带宽（从 BW1 提升到 BW2）来改善算法的性能。

### 矩阵乘法 {#matrix-multiplication}

我们来看看即将成为我们最爱的算法：矩阵乘法（又称 matmul）。给定两个矩阵 $X$ 和 $Y$，我们可以写成 $X * Y \rightarrow Z$，其中 $X$ 的形状为 $\text{bf16}[B, D]$，$Y$ 的形状为 $\text{bf16}[D, F]$，$Z$ 的形状为 $\text{bf16}[B, F]$。要完成这个矩阵乘法，我们需要加载 $2DF + 2BD$ 字节，执行 $2BDF$ FLOPs，并写回 $2BF$ 字节。<d-footnote>严格来说我们执行的是 $BF \times (2D - 1)$ FLOPs，但这已经足够接近了。它来自 $BDF$ 次乘法和 $BF * (D-1)$ 次加法。第 4 节有更多细节。</d-footnote> <d-footnote>虽然矩阵乘法的输出严格来说是 float32，但我们通常会在拷贝回 HBM 之前将其转换为 bfloat16。</d-footnote> 因此：

$$\begin{equation}
\text{Intensity}(\text{matmul}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}
\end{equation}$$

如果假设我们的“批大小（batch size）” $B$ 相对于 $D$ 和 $F$ 很小，就能得到一个很好的简化。于是我们得到

$$\begin{equation}
\frac{BDF}{BD + DF + BF} \approx \frac{BDF}{DF} = B
\end{equation}$$

$$\begin{equation}
\text{Intensity}(\text{matmul}) > \text{Intensity}(\text{TPU}) \implies B > \frac{1.97e14}{8.20e11} = 240
\end{equation}$$

对于 Transformer 中的矩阵乘法，这是一个合理的假设，因为我们通常的局部（每个副本）token 批大小 $B < 1024$（*注意，不是序列数*），而 $D$ 和 $F > 8000$。因此，当每个副本<d-footnote>之所以说“每个副本”，是因为如果我们做某种模型分片来增加参与矩阵乘法的芯片数量，可用的计算能力和内存带宽会按相同比例扩展。因此，临界批大小是针对模型权重的每个独立副本而言的。</d-footnote>的批大小大于 240 个 token 时，我们通常就会变成计算受限——这是一条非常简单的规则！

<p markdown=1 class="takeaway">**要点：** 在大多数 TPU 上，要让 bfloat16 矩阵乘法达到计算受限，我们需要批大小大于 240（个 token）。<d-footnote>注意，这<em>不是</em>通常意义上的批大小——通常意义上的批大小指的是序列数。事实证明，大多数 Roofline 只取决于 token 的数量，与这些 token 属于同一序列还是不同序列无关。例如，如果你在 2048 块 GPU 上使用 512 条序列、每条 4096 个 token 的批大小，那么总批大小就是 `512 * 4096 = 2M` 个 token，局部批大小则是 1k 个 token。</d-footnote></p>

这条规则有几点值得注意的限定条件，我们会在下面的练习题中探讨，尤其是与量化有关的情况（例如，如果我们量化了激活值，但仍然用全精度执行 FLOPs）。不过，这是一条值得记住的好规则。对于 GPU，这个数字略高一些（接近 300），但同样的结论通常也成立。当我们[把一个大的矩阵乘法分解成较小的矩阵乘法](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#your-first-matrix-multiplication-kernel)时，分块（tile）大小也很重要。<d-footnote>在做大型矩阵乘法时，我们需要把它拆分成能放进 VMEM/SMEM/TMEM（带宽更高的片上内存）的较小分块。这会导致某些数据块被多次加载，因此“我们只加载 $O(N^2)$ 字节”这一点就不再完全成立了。考虑一个分块大小为 $bm$、$bk$、$bn$ 的 $(m, k) \cdot (k, n)$ 矩阵乘法。令 $tm = m / bm$，其余类推。那么总 FLOPs 为 $2 \cdot tm \cdot tn \cdot tk \cdot bm \cdot bn \cdot bk$，总字节数为 $2 \cdot tm \cdot tn \cdot (tk \cdot (bm \cdot bk + bk \cdot bn) + bm \cdot bn)$。忽略最后一项，我们得到的算术强度为 $bm \cdot bn / (bm + bn)$，与上面的结果类似。</d-footnote> 我们将在[下一节](../tpus)讨论 GPU 和 TPU 更底层的细节。

### 网络通信的 Roofline {#network-communication-rooflines}

到目前为止我们讨论的所有 Roofline 都是内存带宽 Roofline，*而且都发生在单个芯片内部*。这不应被当作一般规律。事实上，本书中我们关心的大多数 Roofline 都涉及芯片之间的通信：通常是涉及分片到多个 TPU 上的矩阵的矩阵乘法。

举一个有点刻意构造的例子：假设我们要将两个大矩阵 $X\sim \text{bf16}[B, D]$ 和 $Y \sim \text{bf16}[D, F]$ 相乘，它们（沿 $D$ 维度）被均匀地切分到 2 块 TPU/GPU 上。要完成这个乘法（我们将在[第 3 节](../sharding)中看到），我们可以在每块 TPU 上各乘每个矩阵的一半（TPU 0 上计算 `Z0 = X[:, :D // 2] @ Y[:D // 2, :]`，TPU 1 上计算 `Z1 = X[:, D // 2:] @ Y[D // 2:, :]`），然后把得到的“部分和”拷贝到另一块 TPU 上并相加。假设每个方向上每秒可以拷贝 `4.5e10` 字节，每个芯片每秒可以执行 `1.97e14` FLOPs。那么 $T_\text{math}$ 和 $T_\text{comms}$ 分别是多少？

$T_\text{math}$ 显然是之前的一半，因为每块 TPU 只做一半的工作，即<d-footnote>我们忽略了把两个部分和相加所需的 FLOPs（再做 BF 次加法），但这基本可以忽略不计。</d-footnote>

$$T_\text{math} = \frac{2BDF}{2 \cdot \text{Accelerator FLOPs/s}} = \frac{BDF}{1.97e14}$$

那么 $T_\text{comms}$ 呢？它现在指的是芯片之间的通信时间！它就是发送的总字节数除以网络带宽，即

$$T_\text{comms} = \frac{2BF}{\text{Network Bandwidth}} = \frac{2BF}{4.5e10}$$

因此，当 $$\text{Intensity}(\text{matmul (2-chips)}) > \text{Intensity}(\text{TPU w.r.t. inter-chip network})$$ 时，我们就变成计算受限的（现在是相对于芯片间网络而言）；等价地说，当 $\frac{BDF}{2BF} = \frac{D}{2} > \frac{1.97e14}{4.5e10} = 4377$，即 $D > 8755$ 时。注意，与之前不同，现在的临界阈值取决于 $D$ 而不是 $B$！试着想想为什么会这样。这只是一个例子，但我们想强调的是，这类 Roofline 对于判断何时可以把一个操作并行到多个 TPU 上至关重要。

## 几道练习题 {#a-few-problems-to-work}

**问题 1 [int8 矩阵乘法]：** 假设我们想以 int8 精度（每个参数 1 字节）而不是 bfloat16（每个参数 2 字节）来做矩阵乘法 $X[B, D] \cdot_D Y[D, F] \rightarrow Z[B, F]$<d-footnote>在这里以及全书中，我们用记号 $A \cdot_D B$ 表示该乘法在 D 维度上做缩并（contraction）。这是对 einsum 记号的一种不严格用法。</d-footnote>，因为 TPU/GPU 在更低精度下做矩阵乘法更快。

1. 需要从内存中加载多少字节？需要写回内存多少字节？
2. 总共执行了多少次 OPs？
3. 算术强度是多少？
4. $T_\text{math}$ 和 $T_\text{comms}$ 的 Roofline 估计是多少？整个操作运行时间的合理上界和下界是多少？

假设 HBM 带宽为 `8.2e11` bytes/s，int8 峰值 OPs/s 为 `3.94e14`（约为 bfloat16 的 2 倍）。

{% details 点击此处查看答案。 %}

1. 因为参数以 int8 存储，每个参数占 1 字节，所以要从 HBM 加载 $$BD + DF$$ 字节，并写回 $$BF$$ 字节。
2. 这与 bfloat16 下相同，只是理论上 int8 的 OPs/s 应该更快。所以仍然是 $2BDF$ 次 OPs。
3. 算术强度为 $$2BDF / (BD + DF + BF)$$。如果像上面一样假设 $$B \ll D$$ 且 $$B \ll F$$，算术强度就是 $$2B$$，这意味着规则变为 $B > \text{HBM int8 arithmetic intensity} / 2$。代入给定的数字，这个 int8 算术强度为 `3.94e14 / 8.2e11 = 480`，所以规则是 $B > 480 / 2 = 240$。注意这基本没有变化！
4. $$T_\text{math} = 2BDF / 3.94e14$$，$$T_\text{comms} = (BD + DF + BF) / 8.2e11$$，所以一个合理的下界是 $$\max(T_\text{math}, T_\text{comms})$$，上界是 $$T_\text{math} + T_\text{comms}$$。

{% enddetails %}

**问题 2 [int8 + bf16 矩阵乘法]：** 在实践中，我们经常对权重和激活值采用不同的量化方式，因此可能会以很低的精度存储权重，但让激活值（以及计算）保持较高的精度。假设我们想把权重量化为 int8，但激活值（以及计算）保持 bfloat16。批大小达到多少时我们会变成计算受限？假设 bfloat16 为 `1.97e14` FLOPs/s。

*提示：具体来说，这指的是 `bf16[B, D] * int8[D, F] -> bf16[B, F]`，其中 $B$ 是“批大小”。*

{% details 点击此处查看答案。 %}

同样假设 B 很小，我们有 2BDF 次 bfloat16 FLOPs，但权重只有 DF 字节（而 bfloat16 下是 2DF）。这意味着当 $$2B > 240$$，即 $$B > 120$$ 时，我们就会变成计算受限。这个值低了很多，意味着如果我们能做 int8 权重量化（这相当容易做到），同时仍然执行 bfloat16 FLOPs，就能在效率上获得显著的提升（尽管 int8 OPs 会更好）。

{% enddetails %}

**问题 3：** 沿用问题 2 的设定，分别针对 $F = D = 4096$ 和 $F = D = 1024$，画出峰值 FLOPs/s 随 $B$ 变化的 Roofline 图。*使用加载字节数的精确值，而不是近似值。*

{% details 点击此处查看答案。 %}

所求的图如下：

{% include figure.liquid path="assets/img/roofline-plot-q3.png" class="img-fluid img-small" %}

注意，两个模型最终都达到了硬件的峰值 FLOPs/s，但 D/F 较大的那个更早达到。D=F=1024 时的临界批大小几乎翻了一倍。生成这张图的代码如下：

```py
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('batch size')
plt.ylabel('peak bfloat16 FLOPs/s on TPU v5e')
plt.grid()
```

{% enddetails %}

**问题 4：** 如果我们想执行 $\text{int8}[B, D] \cdot_D \text{int8}[B, D, F] \rightarrow \text{int8}[B, F]$，也就是设想每个批元素都有一个不同的矩阵，会怎样？这个操作的算术强度是多少？

{% details 点击此处查看答案。 %}

我们先来看总 FLOPs 和总通信量。

1. 总 FLOPs：FLOPs 基本不变，因为我们在做 $$B$$ 个独立的 $$[D] \times [D, F]$$ 乘积，总工作量与单个 $$[B, D] \times [D, F]$$ 矩阵乘法相同（第 4 节会对此做更多讨论）。所以仍然是 $$2BDF$$。
2. 总通信量：这里的通信量要大得多：$$BD + BDF + BF$$。
3. 因此，我们的算术强度现在实际上是 $$2BDF / (BD + BDF + BF)$$。由于分母中 $$BDF$$ 占主导，这个值大约是 $$2$$。所以它不再依赖于批大小，而基本上是一个常数。这很糟糕，因为这意味着无论如何我们基本上总是通信受限的。

{% enddetails %}

**问题 5 [GPU 的内存 Roofline]：** 利用 [NVIDIA 提供的 H100 SXM 规格表](https://www.nvidia.com/en-us/data-center/h100/)，计算 bfloat16 矩阵乘法在批大小为多少时会变成计算受限。*注意，Tensor Core 的 FLOPs 数值是真实值的两倍，因为它们只有在使用结构化稀疏时才能达到。*

{% details 点击此处查看答案。 %}

从规格表中可以看到，标称的 bfloat16 FLOPs 值为 `1.979e15` FLOPs/s，并带有一个星号注明 “with sparsity”（使用稀疏）。不使用稀疏时的真实值是它的一半，即 `9.89e14` FLOPs/s。内存带宽为 3.35TB/s，即 `3.35e12` bytes / second。因此 $B_\text{crit}$ 为 `9.89e14 / 3.35e12 = 295`，与 TPU 相当接近。

{% enddetails %}

<h3 markdown=1 class="next-section">第 1 部分到此结束！第 2 部分将介绍真实的 TPU 如何处理 FLOPs 和通信，[请点击此处](../tpus)。</h3>
