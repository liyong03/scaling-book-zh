---
layout: distill
title: "你需要知道的所有 Transformer 数学"
# permalink: /main/
description: "这里我们将快速回顾 Transformer 架构，重点是如何计算 FLOPs、字节数以及其他我们关心的量。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 4

previous_section_url: "../sharding"
previous_section_name: "第 3 部分：分片"

next_section_url: ../training
next_section_name: "第 5 部分：训练"

giscus_comments: false
giscus_discussion: 6

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

bibliography: main.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "数点运算"
    id: counting-dots
  - subsections:
    - name: "前向与反向 FLOPs"
      id: forward-and-reverse-flops
  - name: "Transformer 的计算账"
    id: transformer-accounting
  - name: "全局 FLOPs 与参数量计算"
    id: global-flops-and-params-calculation
  - name: "其他数学"
    id: miscellaneous-math
  - subsections:
    - name: "稀疏性与混合专家"
      id: sparsity-and-mixture-of-experts
    - name: "梯度检查点"
      id: gradient-checkpointing
    - name: "键值（KV）缓存"
      id: key-value-kv-caching
  - name: "本节要点是什么？"
    id: what-should-you-take-away-from-this-section
  - name: "几道练习题"
    id: a-few-problems-to-work
  - name: "附录"
    id: appendix
  - subsections:
    - name: "附录 A：Flash Attention 是如何工作的？"
      id: appendix-a-how-does-flash-attention-work

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

## 数点运算 {#counting-dots}

我们从下面这些形状的向量 $$x$$、$$y$$ 和矩阵 $$A$$、$$B$$ 开始：

$$
\def \red#1{\textcolor{red}{#1}}
\def \green#1{\textcolor{green}{#1}}
\def \blue#1{\textcolor{blue}{#1}}
\def \purple#1{\textcolor{purple}{#1}}
\def \orange#1{\textcolor{orange}{#1}}
\def \gray#1{\textcolor{gray}{#1}}

\begin{array}{cc}
\textrm{array}  & \textrm{shape} \\ \hline
x               & \textrm{[P]}   \\
y               & \textrm{[P]}   \\
A               & \textrm{[N P]} \\
B               & \textrm{[P M]} \\
\hline
\end{array}
$$

- 点积 $$x \cdot y$$ 需要 $$P$$ 次*加法*和*乘法*，总共 $$2P$$ 次浮点运算。
- 矩阵-向量乘积 $$Ax$$ 沿 $$A$$ 的各行做 $$N$$ 次点积，共 $$2NP$$ FLOPs。
- 矩阵-矩阵乘积 $$AB$$ 对 $$B$$ 的 $$M$$ 列中的每一列做一次矩阵-向量乘积，总共 $$2NPM$$ FLOPs。
- 一般来说，如果我们有两个更高维的数组 $$C$$ 和 $$D$$，其中一些维度是<span style="color:red">收缩（CONTRACTING）</span>维度，一些是<span style="color:blue">批处理（BATCHING）</span>维度（例如 $$C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$$），那么这次收缩的 FLOPs 开销是 $$C$$ 和 $$D$$ 所有维度之积的两倍，其中批处理维度和收缩维度只计一次（例如 $$2\blue{GH}IJMN\red{KL}$$）。注意，只有当一个维度同时出现在两个乘数中时，它才是批处理维度。（另请注意，如果没有收缩维度、只是逐元素乘积，那么就不需要乘以 2。）<d-footnote><b>收缩</b>维度是在运算过程中被求和掉的轴（它们出现在两个输入中，但不出现在输出中），比如矩阵乘法中的内维度。<b>批处理</b>维度是两个输入共有、并原样保留到输出中的轴；它们索引的是相互独立的子问题，在 FLOP 计数中不会相乘。用 einsum 的话说：同时出现在两个输入和输出中的标签是批处理维度；同时出现在两个输入中但不出现在输出中的标签是收缩维度。</d-footnote>

$$
\begin{array}{ccc}
\textrm{Operation} & \textrm{FLOPs} & \textrm{Data} \\
\hline
x \cdot y  & 2P   & 2P      \\
A x        & 2NP  & NP + P  \\
AB         & 2NPM & NP + PM \\
[c_0,...,c_N] \cdot [d_0,...,d_N] &
2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j
&
  \prod c_i + \prod d_j \\
\hline
\end{array}
$$

请注意，对于矩阵-矩阵乘法，*计算量*按立方 $$O(N^3)$$ 增长，而数据传输量只按平方 $$O(N^2)$$ 增长——这意味着随着矩阵乘法规模的增大，*更容易*达到计算饱和的极限。这是极不寻常的，也在很大程度上解释了为什么我们使用以矩阵乘法为主的架构——它们易于扩展！

{% include figure.liquid path="assets/img/matmul-flops.gif" class="img-fluid" %}

### 前向与反向 FLOPs {#forward-and-reverse-flops}

在训练过程中，我们并不特别关心某次矩阵乘法的结果；我们真正关心的是它的导数。事实证明，计算这个导数的开销大约是做矩阵乘法本身的 3 倍。

假设 **B** 只是一个更大网络中的一个矩阵，**A** 是我们的输入激活值，且 **C = A B**，那么损失 **L** 对 **B** 的导数由链式法则给出：

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)$$

计算它需要 $2NPM$ FLOPs（因为它在 $N$ 维度上做收缩）。同样，损失对 **A** 的导数为

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T$$

由于 **dL/dC** 是一个大小为 $$[N, M]$$ 的矩阵，这同样需要 $2NPM$ FLOPs。虽然这个量不是对某个参数的导数，但它被用来计算网络中前面各层的导数（例如，就像上面用 dL/dC 来计算 dL/dB 一样）。

把这些加起来，我们看到**训练时总共需要 6NPM FLOPs**，而推理时只需要 2NPM：前向传播 2NPM，反向传播 4NPM。由于 PM 是该矩阵的参数量，这就是著名的训练时 Transformer FLOPs 近似公式 $$6 * \text{num parameters} * \text{num tokens}$$ 的最简单形式：每个 token 需要 $$6 * \text{num parameters}$$ FLOPs。我们会在下面给出更准确的推导。

## Transformer 的计算账 {#transformer-accounting}

Transformer 是未来。好吧，至少是现在。也许几年前，它还只是众多架构中的一种。但如今，这一架构的几乎每一个细节都值得了解。我们不会重新介绍这个架构，但[这篇博客](https://jalammar.github.io/illustrated-transformer/)和 [Transformer 原始论文](https://arxiv.org/abs/1706.03762)可能是有用的参考。

下面是 Transformer 解码器架构的基本示意图：

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图：</b>此图展示了标准 Transformer 的一层，从上往下流动。我们用单字母约定来描述 Transformer 中数组的形状和布局，同样用红色表示收缩维度，用蓝色表示批处理维度。在某个给定操作中，输入形状标在左上方，参数形状标在右上方，结果形状标在下方，例如 BTD 是门控 einsum 的输入形状，DF 是权重形状。" %}

**注意 [门控 einsum]**：上图使用了“[门控 einsum](https://arxiv.org/abs/2002.05202)”<d-cite key="glu"></d-cite>，即把上投影矩阵拆成两个矩阵（上图中的 $W_\text{In1}$ 和 $W_\text{In2}$），它们的输出逐元素相乘，起到一种“门控函数”的作用。并不是所有 LLM 都使用这种结构，所以有时你会看到只有一个 $W_\text{In}$ 矩阵，MLP 的总参数量是 2DF 而不是 3DF。通常在这种情况下，D 和 F 会被放大，以使参数量与 3 个矩阵的情形保持一致。话虽如此，LLaMA、DeepSeek 以及许多其他模型都使用了某种形式的门控 einsum。

**注意 2 [MHA 注意力]**：对于自注意力，T 和 S 相同，但对于交叉注意力它们可能不同。对于普通的多头注意力（MHA），N 和 K 相同；对于[多查询注意力](https://arxiv.org/abs/1911.02150)（MQA）<d-cite key="mqa"></d-cite>，K=1；而对于[分组 MQA](https://arxiv.org/abs/2305.13245)（GMQA）<d-cite key="gmqa"></d-cite>，K 只需要整除 N 即可。

**注意 3 [pre-norm 与 post-norm]**：上图展示的是所谓的“pre-norm”架构，其中归一化发生在残差连接之前，通常写作 `x + attn(norm(x))`。如今 LLaMA-3 等模型使用这种结构。Transformer 原始论文使用的是“post-norm”架构，其中 layernorm 发生在残差连接之后，即 `norm(x + attn(x))`。

## 全局 FLOPs 与参数量计算 {#global-flops-and-params-calculation}

我们来计算 Transformer 每层的 FLOPs（这样就不必到处带着 **L** 这个因子）。注意，下面的训练 FLOPs 几乎总是推理 FLOPs 的 3 倍，所以你可以把任何总数除以 3，得到仅前向传播的开销。

### MLP {#mlps}

Transformer 的 MLP 通常由 2 个输入矩阵乘法（其结果逐元素组合）和 1 个输出矩阵乘法组成：

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\[10pt]
A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\[10pt]
\sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\[10pt]
A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\[10pt]
\hline \\
& \approx 18BTDF & 3DF
\end{array}
$$

### 注意力 {#attention}

对于 **Q** 和 **KV** 头数不同的一般分组查询注意力情形，我们假设 **Q**、**K**、**V** 投影的头维度都相同，为 H，并估算 **QKVO** 矩阵乘法的开销：

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\[10pt]
A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\[10pt]
\hline \\ & 12BTD(N+K)H & 2D(N+K)H
\end{array}
$$

点积注意力操作更微妙一些，它实际上是一个在 $$B$$、$$K$$ 维度上做批处理的 $$TH \cdot HS$$ 矩阵乘法，接一个 softmax，再接一个同样在 $$B$$、$$K$$ 维度上做批处理的 $$TS \cdot SH$$ 矩阵乘法。我们用蓝色标出批处理维度：

$$
\begin{array}{cc}
\textrm{operation} & \textrm{train FLOPs} \\
\hline \\[3pt]
Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}]
& 6BTSKGH = 6BTSNH  \\[3pt]
\textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\[3pt]
S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H]
& 6BTSKGH = 6BTSNH \\[3pt]
\hline \\
& \approx 12BTSNH = 12BT^2NH \\
\end{array}
$$

**注意 [因果掩码]**：近来的大多数 Transformer 使用因果掩码，而不是完整的双向注意力。在这种情况下，点积操作的有效 FLOPs 减少一半。要在实践中实现这种减少，我们需要使用注意力 kernel，而不是朴素的 einsum。

### 其他操作 {#other-operations}

Transformer 中还有其他几种操作。Layernorm 相对便宜，在一阶开销估算中可以忽略。注意每层通常有两个 layernorm（一个在注意力之前，一个在 MLP 之前）。此外，最后还有一个巨大的（但不是每层都有的）反嵌入（unembedding）矩阵乘法。

$$
\begin{array}{ccc}
\textsf{operation} & \textsf{train FLOPs} & \textsf{params} \\
\hline \\
2 \times \textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{2D} \\[10pt]
A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\
\end{array}
$$

### Transformer FLOPs 的通用经验法则 {#general-rule-of-thumb-for-transformer-flops}

如果我们忽略点积注意力的开销（对于较短上下文的训练，这是合理的），那么所有层的总 FLOPs 为

$$
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{num tokens} * \textrm{parameter count}
\end{align*}
$$

这就得到了一个著名的经验法则，用于估算稠密 Transformer 的 FLOP 数（忽略注意力 FLOPs）。（反嵌入是另一个简单的矩阵乘法，有 $6BTDV$ FLOPs 和 $DV$ 个参数，同样遵循这一经验法则。）

### 注意力开销占比随上下文长度的变化 {#fractional-cost-of-attention-with-context-length}

如果我们把上面的点积注意力也计算在内，并假设 $$F=4D$$、$$D=NH$$（这是典型情况）以及 $$N=K$$，那么点积注意力 FLOPs 与所有矩阵乘法 FLOPs（包括注意力投影）之比为：

$$\small{\frac{\textrm{attention FLOPs}}{\textrm{matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}$$

结论是，**在训练中，只有当 T>8D 时，点积注意力 FLOPs 才会占主导**。对于 D ~ 8k，这大约是 64K 个 token。这在一定程度上是合理的，因为它意味着随着 MLP 规模增大，注意力 FLOPs 变得不那么关键。对于大模型来说，注意力的平方开销实际上并不是更长上下文训练的巨大障碍。然而，对于较小的模型，例如 D=4608 的 Gemma-27B，注意力在序列长度约 37k 时就会占主导。<d-footnote>注意，一些现代开源模型引入了局部注意力或其他优化，降低了注意力的开销，也改变了这里的 Roofline。</d-footnote> Flash Attention 也有助于缓解长上下文的开销，我们会[在附录 A 中](#appendix-a-how-does-flash-attention-work)简要讨论。

## 其他数学 {#miscellaneous-math}

### 稀疏性与混合专家 {#sparsity-and-mixture-of-experts}

如果不简要讨论一下混合专家（MoE）模型<d-cite key="moe"></d-cite>，那就是我们的疏忽了。MoE 模型把标准 Transformer 中单个稠密 MLP 块替换为一组相互独立、可以动态路由的 MLP。初步近似地说，**MoE 就是每层有 E 个 MLP 块（而不是只有一个）的普通稠密模型**。每个 token 激活其中 $k$ 个专家，通常 $k \ll E$。比值 $E / k$ 被称为稀疏度，通常在 8 到 64 之间（例如 [DeepSeek v3](https://arxiv.org/pdf/2412.19437) 实际上是 $k=8$、$E=256$）。与稠密版本相比，这使参数量变为原来的 $O(E)$ 倍，而每个 token 激活的总参数量变为原来的 $k$ 倍。

{% include figure.liquid path="assets/img/moe.png" class="img-fluid img-small" caption="<b>图：</b>一个有 $n$ 个专家的 MoE 层示例。门控专家把每个 token 路由到其中 $k$ 个专家，这 $k$ 个 MLP 的输出会被求和。参数量是每个专家大小的 $n$ 倍，但每个 token 只用到其中 $k$ 个。<a href=\"https://deepgram.com/learn/mixture-of-experts-ml-model-guide\">来源</a>。" %}

与稠密模型相比，MoE 引入了新的通信，主要是两次 AllToAll（一次在 MoE 块之前，一次在之后），用于把 token 路由到正确的专家，再把它们送回原来的设备。<d-footnote>严格来说，只有当我们沿着与专家相同的轴做数据分片或序列分片时，才会发生这种情况。</d-footnote>不过，正如我们在上一节看到的，每次 AllToAll 的开销只有沿单个轴进行的同等 AllGather 的 1/4（对于双向环）。

### 梯度检查点 {#gradient-checkpointing}

反向传播作为一种算法，是用内存换计算。反向传播不需要 $$O(n_\text{layers}^2)$$ FLOPs，**而是需要 $$O(n_\text{layers})$$ 的内存**，保存前向传播中生成的所有中间激活值。虽然这比平方级计算要好，但它在内存方面极其昂贵：一个 $$B * T=4M$$（每个批次共 4M 个 token）、L=64、D=8192 的模型，如果要避免所有不必要的反向传播计算，就需要以 bfloat16 保存大约 $$2 * 20 * B * T * D * L = 84TB$$ 的激活值。这里的 20 来自（粗略地）统计上面 Transformer 示意图中的每个中间节点，因为例如

$$f(x) = \exp(g(x))$$

$$\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}$$

所以为了避免重新计算，我们需要保存前向传播中的 $$g(x)$$ 和 $$\exp(g(x))$$。为了避免占用这么多内存，我们可以选择只保存一部分中间激活值。下面是我们使用的几种策略。

* **块重计算（Block remat）**：只保存每层的输入。这是我们使用的最激进的方法，每层只保存 1 个检查点，这意味着在上面的例子中我们只需保存 4.2TB。这迫使我们在反向传播中重复几乎所有的前向传播 FLOPs，即 FLOPs 从 $$6 \cdot \text{num params} \cdot \text{num tokens}$$ 增加到大约 $$8 \cdot \text{num params} \cdot \text{num tokens}$$。
* **只保存大矩阵乘法**：另一个简单的策略是只保存大矩阵乘法的输出。这让我们在反向传播中避免重新计算任何大矩阵乘法，但仍需要重新计算其他激活函数和注意力的部分内容。这把上面每层 20 个减少到接近每层 7 个。

这些远不是全部。使用 JAX 时，这些通常由 `jax.remat`/`jax.checkpoint` 控制（你可以在[这里](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)了解更多）。

### 键值（KV）缓存 {#key-value-kv-caching}

正如我们将在[第 7 节](../inference)中看到的，LLM 推理有两个关键部分：预填充（prefill）和生成。

* **预填充**处理一个长提示词，并把它的注意力激活值保存在键值缓存（KV 缓存）中供生成时使用，具体来说是注意力块中的键值投影。
* **生成**把若干个这样的 KV 缓存批处理在一起，并从每个缓存中采样 token。

于是每个 KV 缓存实际上是一个大小为 $[2, S, L, K, H]$ 的数组，其中 2 对应键和值。这相当大！以 int8 存储时，键值缓存的总大小为 $2SLKH$。对于一个中等规模的模型，上下文长度为 8k、64 层、$KH = NH = D = 8192$，其大小为 $2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$。你可以看出为什么我们会希望使用 $K \ll N$ 的 GMQA。

## 本节要点是什么？ {#what-should-you-take-away-from-this-section}

* Transformer 的总参数量和 FLOPs 相当容易计算，这里假设使用 MHA（批大小为 B，词表大小为 V，序列长度为 T，D=d<sub>model</sub>，F=d<sub>ff</sub>），总结如下：


<!-- $$
\begin{array}{ccc}
\textrm{Component} & \textrm{Params per layer} & \textrm{Training FLOPs per layer} \\
\hline \\
\textbf{MLP} & 3DF & 18BTDF \\[10pt]
\textbf{Attention} & 4DNH & 24BTDNH + 12BT^2NH \\[10pt]
\textbf{Other} & D & BTD \\[10pt]
\textbf{Vocab} & DB \text{ (total, not per-layer)} & 12BTDV \\[10pt]
\end{array}
$$ -->


| 组件          | 每层参数量                | 每层训练 FLOPs                |
| :------------ | :------------------------ | :---------------------------- |
| **MLP**       | 3DF                       | 18BTDF                        |
| **注意力**    | 4DNH                      | 24BTDNH \+ 12BT<sup>2</sup>NH |
| **其他**      | 2D                        | BTD                           |
| **词表**      | DV（总计，不是每层）      | 12BTDV                        |

* MLP 块的参数量在总参数量中占主导，而且只要序列长度 $T < 8D$，MLP 块在 FLOPs 预算中也占主导。
* 对于合理的上下文长度，训练期间的总 FLOPs 预算可以很好地用 $$6 \cdot \text{num_params} \cdot \text{num_tokens}$$ 来近似。
* 在推理期间，每个 KV 缓存大约为 $$2 \cdot S \cdot L \cdot K \cdot H$$（其中 K 是 KV 头的数量），不过架构上的改动通常可以降低这一数值。

## 几道练习题 {#a-few-problems-to-work}

**问题 1：** 一个 $D=4096$、$F=4 \cdot D$、$V=32,000$、$L=64$ 的模型有多少参数？其中注意力参数占多大比例？每个 token 的 KV 缓存有多大？*你可以假设 $N\cdot H=D$，并且使用多头注意力、KV 为 int8。*

{% details 点击此处查看答案。 %}

1. 总参数量大约为 $$L \cdot (3DF + 4DNH + 2D) + 2DV$$（计入每层的两个 layernorm）。代入给定数值，得到 $$64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 2 \cdot 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$$，即 16B 参数。
2. 一般来说，注意力参数占总参数的比例为 $$4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$$。也就是说，大约 1/4 的参数用在注意力上。
3. 每个 token 的 KV 缓存为 $$2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$$ 个 int8 值，即 `512 KiB / token`。

{% enddetails %}

**问题 2：** 在 `{'X': 4, 'Y': 8, 'Z': 4}` 上执行 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] 总共需要多少 FLOPs？每个 TPU 执行多少 FLOPs？

{% details 点击此处查看答案。 %}

该操作“理论上”的总 FLOPs 为 $$2 \cdot B \cdot D \cdot F$$。然而，由于计算没有沿 Z 维度分片，我们实际上多做了 Z 倍的 FLOPs，即总 FLOPs 为 $$2 \cdot B \cdot D \cdot F \cdot Z$$。由于计算沿其他维度是分片的，每个设备上的 FLOPs 大约为 $$2 \cdot B \cdot D \cdot F / (X \cdot  Y)$$。

{% enddetails %}

**问题 3：** 执行 $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$ 涉及多少 FLOPs？

{% details 点击此处查看答案。 %}

按照上面的规则，I 和 J 是收缩维度，K、L、M、N 和 O 是非收缩维度。这里没有“批处理维度”，所以结果就是 $$2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$$，即所有轴大小的乘积。如果有共享轴，它只会被计算一次。

{% enddetails %}

**问题 4：** 分组多查询注意力（grouped multi-query attention）的算术强度是多少（忽略 Q/K/V/O 投影）？*请将答案表示为 Q 和 KV 序列长度 T、S 以及多查询因子 G 的函数。* 上下文长度达到多少时注意力会变成计算受限？根据我们 TPU 的 HBM 带宽，画出随上下文长度增长，注意力相对于 FFW 块的有效相对开销。*提示：假设我们使用的是高效的注意力实现，不做任何不必要的读写。分别考虑 T = S 和 T << S 这两种极限情况。*

{% details 点击此处查看答案。 %}

自注意力需要加载 $$Q$$、$$K$$ 和 $$V$$ 激活值，然后计算 $$\text{softmax}(Q \cdot K) \cdot V$$，再把结果写回 HBM。实际上这会用 Flash Attention 来实现，所以这里的推算有一些需要注意的地方，但基本上，bf16 下的自注意力执行的是：

$$\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}$$

$$U=\text{softmax}_S(\text{O[B, T, S, K, G]})$$

$$\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}$$

因此总字节数为 $$2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$$，总 FLOPs 为 $$4BTSNH + O(BTSN)$$，算术强度为 $$4BTSKGH / (4BHK * (TG + S))$$。

所以基本上，在预填充阶段有 $$S=T$$，算术强度为 $$4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$$。在生成阶段 $$T=1$$，于是有 $$4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$$（假设 $$S$$ 非常大）。取决于你如何理解这个问题，在不做序列分片的前提下，预填充或训练期间的自注意力在 S=240 时达到计算受限。在生成阶段，由于 $$G$$ 很小，我们永远不会是计算受限的。尽管如此，你可以看到增大 $$G$$ 会让我们更接近计算受限。

{% enddetails %}

**问题 5：** 序列长度为多少时，自注意力的 FLOPs 等于 QKVO 投影的 FLOPs？

{% details 点击此处查看答案。 %}

这纯粹就是求 $$24BTDNH = 12BT^2NH$$ 何时成立。化简得到 $$2D = T$$，所以例如当 $$D=4096$$ 时，这个值是 $$8192$$。这说明对于大多数合理的上下文长度，矩阵乘法的 FLOPs 更大。

{% enddetails %}

**问题 6：** 假设在前向传播中我们只保存 Transformer 层中 7 个主要矩阵乘法的输出（Q、K、V、O \+ 三个 FFW 矩阵）。在反向传播中我们需要额外多少 FLOPs 来进行“重计算”？

{% details 点击此处查看答案。 %}

只保存七个矩阵乘法的输出（Q、K、V、O、W₁、W₂、W₃）意味着反向传播必须重新计算两个注意力矩阵乘法

$$QK^{\top} \quad\text{and}\quad \operatorname{softmax}(QK^{\top})V$$

才能得到 $\frac{\partial L}{\partial W_\text{O}}$。

每个都是在 $B$ 条序列和 $N$ 个头上批处理的 $T \times T$ 矩阵乘法，所以额外的 FLOPs 为

$$4 \; B \, T^{2} \, N \, H.$$

其他需要重计算的操作有：
1. 为得到 $\frac{\partial L}{\partial W_\text{In1}}$ 和 $\frac{\partial L}{\partial W_\text{In2}}$ 需要 $O(BTD)$。
2. 以及为得到 $\frac{\partial L}{\partial W_\text{Out}}$ 需要 $O(BTF)$。

{% enddetails %}

**问题 7：** DeepSeek v3 称其在 14.8T token 上训练了 2.79M H800 小时（[来源](https://arxiv.org/pdf/2412.19437v1)）。已知它有 37B 激活参数，他们大致达到了多少硬件利用率？*提示：注意他们使用的是不带结构化稀疏的 FP8 FLOPs。*

{% details 点击此处查看答案。 %}

从[这里](https://lenovopress.lenovo.com/lp1814.pdf)的规格表中，我们查到带稀疏时的 FP8 性能为 3,026 TFLOPs/s，不带稀疏时通常是其一半（`1.513e15` FLOPs/s）。2.79M H800 小时意味着总共 `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25` FLOPs。考虑到 37B 的激活参数量，这次训练应该使用了大约 `6 * 37e9 * 14.8e12 = 3.3e24` FLOPs。这意味着 FLOPs 利用率大约为 `3.3e24 / 1.52e25 = 21.7%`。

{% enddetails %}

**问题 8：** 混合专家（MoE）模型有 $E$ 个标准稠密 MLP 块的副本，每个 token 激活其中 $k$ 个专家。对于权重为 int8 的 MoE，在 TPU v5e 上需要多大的批大小（以 token 计）才能达到计算受限？对于有 256 个（路由）专家且 $k=8$ 的 DeepSeek，这个数是多少？

{% details 点击此处查看答案。 %}

由于每个专家有 $E$ 个副本，在 int8 下，每个权重矩阵需要加载 $E \cdot D \cdot F$ 字节。由于每个 token 激活 $k$ 个专家，每个权重矩阵对应 $2\cdot k \cdot B \cdot D \cdot F$ FLOPs。要在 int8 权重、bfloat16 FLOPs 的情况下达到计算受限，我们需要算术强度（每加载一个字节对应的 FLOPs）超过 TPU 约 240 FLOPs/byte 的临界值，即 $(2\cdot k \cdot BDF) / EDF > 240$，或 $k \cdot B / E > 120$。

因此，我们需要 $B > 120 \cdot E / k$ 才能达到计算受限。对于 DeepSeek，这给出 $B > 120 \cdot 256 / 8 = 3840$。对于生成阶段来说，这是一个相当大的批大小。

{% enddetails %}

<h3 markdown=1 class="next-section">第 4 部分到此结束！第 5 部分（关于扩展 Transformer 训练）请[点击此处](../training)！</h3>

## 附录 {#appendix}

### 附录 A：Flash Attention 是如何工作的？ {#appendix-a-how-does-flash-attention-work}

反对将 Transformer 扩展到超长上下文的传统理由是：注意力的 FLOPs 和内存占用随上下文长度呈二次方增长。虽然注意力的 QK 乘积确实具有形状 $[B, T, S, N]$（其中 B 是批大小，T 和 S 是 Q 和 K 的序列维度，N 是头数），但这一说法有几个重要的前提需要注意：

1. 正如我们前面指出的，尽管它是二次方的，但只有当 $$T > 8 \cdot D$$ 时注意力 FLOPs 才会占主导；而且在训练期间，与驻留在内存中的所有权重和激活值检查点相比，单个注意力矩阵占用的内存很小，尤其是在分片之后。
2. 我们并不需要把完整的注意力矩阵物化出来才能计算注意力！我们可以计算局部的和与最大值，从而在任何时候都只需物化数组中的一小块。虽然总 FLOPs 仍是二次方的，但我们大幅降低了内存压力。

第二点最早由 [Rabe et al. 2021](https://arxiv.org/abs/2112.05682) 提出，后来又出现在 [Flash Attention 论文](https://arxiv.org/abs/2205.14135)（Dao et al. 2022）中。基本思路是按 K/V 的块来计算注意力：我们计算局部 softmax 和一些辅助统计量，然后把它们传给下一个块，由下一个块将其与自己的局部结果合并。具体来说，我们计算：

1. **M：** $$q \cdot k$$ 沿序列维度的累积最大值
2. **O：** 沿序列维度的累积完整注意力 softmax
3. **L：** 累积分母 $$\sum_i \exp(q \cdot k_i - \text{running max})$$

有了这些，我们只需常数量级的内存就能计算新的最大值、新的累积和以及新的输出。粗略地描述一下其工作原理，注意力大致是这样的操作：

$$\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}$$

减去最大值是为了数值稳定性，而且由于 $$\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$$，减去它不会影响结果。只看上式的分母，设想我们有两个相邻的键向量块 $$K^1$$ 和 $$K^2$$，并分别计算它们的局部 softmax 和 $$L^1$$ 与 $$L^2$$：

$$L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)$$

$$L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)$$

那么我们可以用下式将它们合并为这两个块整体的完整 softmax 和：

$$L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2$$

其中

$$M^1 = \max_j Q \cdot K_j^1 \text{ and } M^2 = \max_j Q \cdot K_j^2$$

对完整的 softmax 也可以这样做，这样我们就能累加任意大的 softmax 和。下面是 Flash Attention 论文中的完整算法。

{% include figure.liquid path="assets/img/flash-algo.png" class="img-fluid" %}

从硬件角度看，这让我们可以把 Q 的块放进 VMEM（即上面算法中所说的片上 SRAM），这样每次迭代只需加载 KV 块，从而提高了算术强度。我们也可以把累积统计量保存在 VMEM 中。

最后一个值得强调的微妙之处是注意力 softmax 的一个性质，正是它让 Flash 的 VJP（反向模式导数）计算在训练中变得可行。我们定义一个中间 softmax 数组：

$$S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_l e^{\tau q_i \cdot k_l}}$$

在注意力中，我们由反向模式的 *dO* 和 *V* 数组得到 *dS*：

$$dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}$$

在把这个梯度反向传播到 Q 和 K 的过程中

$$d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}$$

我们利用一个恒等式，把沿较大的键**长度**维度的收缩，替换为沿特征**深度**维度的局部收缩。

$$\begin{align*}
S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\
&= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\
&= \sum_d dO_{id} O_{id} \\
&= dO_{id} \cdot_d O_{id}
\end{align*}$$

这一替换对于实现按序列分块的*局部* VJP 计算至关重要，并使环形注意力（ring attention）等更巧妙的分片方案成为可能。
