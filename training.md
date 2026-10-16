---
layout: distill
title: "如何为训练并行化 Transformer"
# permalink: /main/
description: "本章讨论 LLM 训练中使用的四种主要并行方案：数据并行、完全分片数据并行（FSDP）、张量并行和流水线并行。对于每一种方案，我们都会计算在什么情况下会遇到通信瓶颈。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 5

previous_section_url: "../transformers"
previous_section_name: "第 4 部分：Transformer 数学"

next_section_url: ../applied-training
next_section_name: "第 6 部分：训练 LLaMA"

bibliography: main.bib

giscus_comments: false
giscus_discussion: 7

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
  - name: "我们所说的扩展是什么意思？"
    id: what-do-we-mean-by-scaling
  - subsections:
    - name: "数据并行"
      id: data-parallelism
    - name: "完全分片数据并行（FSDP）"
      id: fully-sharded-data-parallelism-fsdp
    - name: "张量并行"
      id: tensor-parallelism
    - name: "结合 FSDP 与张量并行"
      id: combining-fsdp-and-tensor-parallelism
    - name: "流水线"
      id: pipelining
    - name: "跨 Pod 扩展"
      id: scaling-across-pods
  - name: "TPU 上 LLM 训练的要点"
    id: takeaways-from-llm-training-on-tpus
  - name: "练习题"
    id: some-problems-to-work
  - name: "附录"
    id: appendix
  - subsections:
    - name: "附录 A：推导反向传播的通信"
      id: appendix-a-deriving-the-backward-pass-comms

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

## 我们所说的扩展是什么意思？ {#what-do-we-mean-by-scaling}

“模型扩展”的目标是：在增加用于训练或推理的芯片数量的同时，让吞吐量随之成比例地线性增长（我们称之为*强扩展*，strong scaling）。单芯片上的性能取决于内存带宽与 FLOPs 之间的权衡，而集群层面的性能则取决于能否把芯片间通信与有用的 FLOPs 重叠起来，从而将通信隐藏掉。这并不简单，因为增加芯片数量会加大通信负载，同时又减少了每个设备上可用来隐藏通信的计算量。正如我们在[第 3 部分](../sharding)中看到的，分片矩阵乘法往往需要代价高昂的 AllGather 或 ReduceScatter，它们可能会阻塞 TPU，使其无法进行有用的工作。本节的目标就是弄清楚这些操作在什么时候会变得*过于昂贵*。

本节将讨论四种常见的并行方案：（纯）**数据并行、完全分片数据并行**（FSDP / ZeRO 分片）、**张量并行**（也称模型并行），以及（简要介绍的）**流水线并行**。对于每一种方案，我们都会说明它会带来多少通信开销，以及这一开销在什么情况下会开始成为计算开销的瓶颈。<d-footnote>我们将重点关注通信方面的限制——虽然内存容量的约束也很重要，但在预训练中使用重计算（rematerialization，即激活值检查点）以及非常多芯片的情况下，它通常不会成为限制因素。我们在这里也不讨论 MoE 的专家并行——它会大幅扩展设计空间——只讨论稠密 Transformer 这一基本情形。</d-footnote>在本节中，你可以只关注芯片间的通信开销，因为只要单芯片的批大小足够大，从 HBM 到 MXU 的数据传输就已经与计算重叠了。

为了简化计算，本节将使用以下记号。

| 记号 | 含义（模型参数）                                             |
| :------- | :--------------------------------------------------------------------- |
| D        | **d**<sub>model</sub>（隐藏维度 / 残差流维度）      |
| F        | **d**<sub>ff</sub>（前馈层维度）                        |
| B        | 批维度（批中的 token 数；指总数，而非每个设备上的数量） |
| T        | 序列长度                                                        |
| L        | 模型的层数                                          |

| 记号 | 含义（硬件特性）                                                                 |
| :------- | :------------------------------------------------------------------------------------------------ |
| C        | 每芯片的 FLOPS/s                                                                                  |
| W        | 网络带宽（双向，常带下标，例如 $W_{\text{ici}}$ 或 $W_{\text{dcn}}$） |
| X        | 沿网格轴 X 的芯片数                                                                 |
| Y        | 沿另一条网格轴（记为 Y）的芯片数                                           |
| Z        | 沿第三条网格轴（记为 Z）的芯片数                                                |

为简单起见，**我们把 Transformer 近似为一摞 MLP 块**——正如我们在[第 4 部分](../transformers)中看到的，对于较大的模型，注意力在 FLOPs 中所占的比例相对较小。我们还会忽略门控矩阵乘法，这样每一层就只剩下如下的简单结构：

{% include figure.liquid path="assets/img/transformer-layer.png" class="img-fluid" caption="<b>图：</b>一个简化的 Transformer 层。我们把每个 FFW 块看作两个矩阵的堆叠：<b>W<sub>in</sub></b>: <code>bf16[D, F]</code>（上投影）和 <b>W<sub>out</sub></b>: <code>bf16[F, D]</code>（下投影），输入为 <b>In</b>: <code>bf16[B, D]</code>。" %}

{% details 下面是我们这个不带任何并行的小 Transformer 的完整算法。 %}

<div markdown=1 class="algorithm">

**前向传播：**需要计算 Loss[B]

1.  Tmp[B, F] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B, D] = Tmp[B, F] *<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B] = ...

**反向传播：**需要计算 dW<sub>out</sub>[F, D]、dW<sub>in</sub>[D, F]

1.  dOut[B, D] = ...
2.  dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
3.  dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
4.  dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
5.  dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前面的层需要用到*)

</div>

给出这个算法是为了与加入通信之后的算法做对比。

{% enddetails %}

下面是我们将要讨论的 4 种并行方案。每种方案都可以看作由上图中 **In**、**W<sub>in</sub>、W<sub>out</sub> 和 Out** 的分片方式唯一确定。

**1. 数据并行：** *激活值沿批维度分片，参数和优化器状态在每个设备上复制一份。通信只发生在反向传播期间。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**2. 完全分片数据并行（FSDP 或 ZeRO-3）：** *激活值沿批维度分片（与纯数据并行相同），参数沿同一条网格轴分片，并在前向传播中使用之前即时（just-in-time）进行 AllGather。优化器状态同样沿批维度分片。这减少了重复的内存占用。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

**3. 张量并行（也称 Megatron 分片或模型并行）：** *激活值沿 D（$d_\text{model}$）分片，参数沿 F（$d_{ff}$）分片。在每个块之前和之后分别对激活值做 AllGather 和 ReduceScatter。可与 FSDP 兼容。*

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

**4. 流水线并行：** *权重沿层维度分片，激活值被划分为微批次，并沿层维度依次传递。流水线各阶段之间的通信量极小（只需将激活值移动一跳）。稍微滥用一下记号：*

$$\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]$$

### 数据并行 {#data-parallelism}

**语法：** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

当你的模型即使在很小的批大小下（>240 个 token，以保证计算受限）也能放进单个芯片时，**你应当始终使用简单的数据并行。**只要 TPU 数量小于批大小，纯数据并行就可以把激活值切分到任意数量的 TPU 上。前向传播不涉及任何通信，但在每一步结束时，**每个 TPU 都要对其本地梯度执行一次 AllReduce，在更新参数之前将梯度同步。**

{% include figure.liquid path="assets/img/data-parallelism.png" class="img-fluid" caption="<b>图：</b>纯数据并行（前向传播）示意图。激活值（左）沿批维度完全分片，而权重则完全复制，因此每个 TPU 都拥有一份相同的权重副本。这意味着权重占用的总内存增加为原来的 N 倍，但前向传播中不需要任何通信。" %}

{% details 下面是前向和反向传播的完整算法。为了简洁，我们滥用记号，把 dL/dOut 写作 dOut。 %}

<div markdown=1 class="algorithm">

**纯数据并行算法：**

**前向传播：**需要计算 Loss[B<sub>X</sub>]

1.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B<sub>X</sub>] = ...

**反向传播：**需要计算 dW<sub>out</sub>[F, D]、dW<sub>in</sub>[D, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上，可以异步执行*)
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
5.  dW<sub>in</sub>[D, F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) (*不在关键路径上，可以异步执行*)
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前面的层需要用到*)

</div>

我们忽略损失函数的细节，并简记 $\text{Tmp} = W_\text{in} \cdot \text{In}$。注意，虽然最终的损失是平均后的 **AllReduce**(Loss[B<sub>X</sub>])，但我们只需要在反向传播中对权重梯度求平均时执行 AllReduce。

{% enddetails %}

注意，前向传播中没有任何通信——**通信全部发生在反向传播中**！反向传播还有一个很好的性质：这些 AllReduce 不在“关键路径”上，也就是说每个 AllReduce 都可以在任何方便的时候执行，不会阻塞后续操作。如果总通信开销超过了总计算开销，它*仍然可能成为我们的瓶颈*，但从实现的角度来看，这要宽松得多。我们会看到，模型并行 / 张量并行并不具备这一性质。

**为什么要这样做？**纯数据并行通过在批维度上切分激活值来减轻激活值的内存压力，只要有更多芯片来切分批维度，我们几乎可以任意增大批大小。尤其是在训练期间，激活值往往占据了大部分内存，这一点非常有帮助。

**为什么不这样做？**纯数据并行完全无法减轻模型参数或优化器状态带来的内存压力。这意味着，对于参数 + 优化器状态放不进单个 TPU 的、规模较大的有意义模型，纯数据并行很少有用武之地。为了给出一个量级上的直观感受：如果我们用 bf16 存储参数、用 fp32 存储优化器状态并使用 Adam 训练<d-footnote>Adam 需要存储参数以及一阶和二阶累加量。由于参数为 bfloat16、优化器状态为 float32，每个参数需要 `2 + 8 = 10` 字节。</d-footnote>，那么能放下的最大模型参数量为 $$\text{TPU 内存} / 10$$，例如在一块拥有 96GB HBM 的 TPUv5p 芯片上使用纯数据并行，大约是 9B 参数。

<p markdown=1 class="takeaway">**要点**：使用 Adam 和纯数据并行时，我们能训练的最大模型满足 $$\text{num_params} = \text{每设备 HBM} / 10$$。对于 TPU v5p，大约是 9B 参数。<d-footnote>注意，这里没有计入梯度检查点，因此实际上并不可用。这是在批大小仅为 1 个 token 时的绝对下界。</d-footnote></p>

*为了让它在训练真实模型时派上用场，我们至少需要对模型参数或优化器进行部分分片。*

**我们什么时候会遇到通信瓶颈？**如上所示，每层有两次 AllReduce，每次的大小为 $$2DF$$（对于 bf16 权重）。那么数据并行什么时候会让我们变成通信受限呢？

与上表一样，令 $C$ = 每芯片 FLOPs，$W_{\text{ici}}$ = **双向**网络带宽，$X$ = 批被划分成的分片数<d-footnote>我们假设这一划分是在 ICI 网格上完成的，因此相关的网络带宽是 $W_\text{ici}$</d-footnote>。我们来计算执行相关矩阵乘法所需的时间 $$T_\text{math}$$，以及所需的通信时间 $$T_\text{comms}$$。由于这种并行方案在前向传播中不需要通信，我们只需要为反向传播计算这些量。

*通信时间：*从前面的章节我们知道，在一维网格中执行一次 AllReduce 所需的时间只取决于被 AllReduce 的数组的总字节数以及 ICI 带宽 $W_\text{ici}$；具体来说，AllReduce 时间为 $2 \cdot \text{总字节数} / W_\text{ici}$。由于需要对 $W_\text{in}$ 和 $W_\text{out}$ 都做 AllReduce，每层有 2 次 AllReduce。每次 AllReduce 针对的是一个权重矩阵，即一个包含 $DF$ 个参数、也就是 $2DF$ 字节的数组。综合起来，单层中 AllReduce 的总时间为

$$\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}$$

*矩阵乘法时间：*每层在前向传播中包含两次矩阵乘法，在反向传播中包含四次矩阵乘法，每次都需要 $2(B/X)DF$ FLOPs。因此，对于反向传播中的单层，我们有

$$\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}$$

由于两者是重叠的，每层的总时间就是这两个量中的较大值：

$$\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}$$

当 $$T_\text{math}/T_\text{comms} > 1$$ 时，我们就是计算受限的，即当

$$\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}$$

结论是：要在数据并行下保持计算受限，每个设备上的批大小 $$B / X$$ 必须超过 ICI 运算强度 $C / W_\text{ici}$。归根结底，这是因为计算时间随每设备批大小增长，而通信时间与这个量无关（因为我们传输的是模型权重）。注意 $B/X > C/W_\text{ici}$ 这个条件与单设备计算受限规则 $B > 240$ 的相似之处；在那种情况下，这条规则同样源于计算时间随批大小增长、而数据传输量（在 $B \ll F, D$ 的情形下）与批大小无关这一事实。

我们代入一些真实数字来感受一下量级。对于 TPUv5p，在 ICI 上做一维数据并行时 `C=4.6e14`、`W=2 * 9e10`，因此**每个芯片上的批大小至少要达到 2,550，才能避免通信受限**。由于数据并行可以跨多条轴进行，如果我们把 TPUv5p Pod 的全部三条轴都用于纯数据并行，带宽 $W_\text{ici}$ 就会变为 3 倍，从而可以把每个 TPU 的批大小缩小到仅 BS=850，即每个 Pod（8960 个芯片）每批 7.6M 个 token！**这说明纯数据并行其实相当不容易成为瓶颈！**

<p markdown=1 class="takeaway">**注意［上下文并行］：**在本节中，$B$ 始终指以 **token 为单位**的总批大小。但显然，我们的批是由许多不同的序列组成的，那这是怎么回事呢？对于 MLP 来说，**token 就是 token**！它们属于同一个序列还是两个不同的序列都无关紧要。因此，我们或多或少可以自由地同时在批维度和序列维度上做数据并行：我们称之为上下文并行或序列并行，但你可以把它简单地看作另一种数据并行。注意力比 MLP 要棘手一些，因为它涉及跨序列的计算，但这可以通过在注意力计算中 gather KV 或 Q，并小心地重叠 FLOPs 与通信来处理（通常使用一种叫做“ring attention”的方法）。在本节中，我们将完全忽略序列维度，并假设采用了一定程度的批并行或序列并行。</p>

**关于多条网格轴的说明：**这里简要说明一下多条轴如何影响可用带宽。当我们为某种并行策略使用多条网格轴时，就能获得更多带宽。

* **定义：**$M_X$（$M_Y$、$M_Z$ 等）是某种并行策略所跨越的硬件网格轴的数量。
* **效果（带宽受限时）：**使用 $M$ 条轴可以提供（$\approx M$ 倍的）聚合链路带宽，因此集合通信时间按 $\propto 1/M_X$ 缩放。

### 完全分片数据并行（FSDP） {#fully-sharded-data-parallelism-fsdp}

**语法：** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

完全分片数据并行（通常称为 FSDP 或 ZeRO 分片<d-cite key="zero"></d-cite>）把模型的优化器状态和权重切分到各个数据并行分片上，并在需要时高效地对它们进行 gather 和 scatter。**与纯数据并行相比，FSDP 大幅降低了每个设备的内存占用，并节省了反向传播的 FLOPs，而额外开销极小。**

{% include figure.liquid path="assets/img/fsdp.png" class="img-fluid" caption="<b>图：</b>FSDP 沿数据维度对 Win 的收缩维度和 Wout 的输出维度进行分片。这减少了内存占用，但（根据第 3 部分）要求我们在执行矩阵乘法之前先 gather 权重 W。注意，激活值（左）<i>没有沿收缩维度分片</i>，正是这一点迫使我们进行 gather。<b>注意，权重的优化器状态同样沿收缩维度分片。</b>" %}

你应该还记得（见[第 3 部分](../sharding)），一次 AllReduce 可以分解为一次 AllGather 和一次 ReduceScatter。这意味着，我们不必像标准数据并行那样对梯度做完整的 AllReduce，而是可以把权重和优化器状态分片到各个芯片上，在前向传播的每一层对它们做 AllGather，并在反向传播中对权重（梯度）做 ReduceScatter，且不产生额外开销。

{% details 下面是 FSDP 的完整算法。 %}

<div markdown=1 class="algorithm">

**完全分片数据并行（FSDP）：**

**前向传播：**需要计算 Loss[B<sub>X</sub>]

1.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*不在关键路径上，可以在上一层期间完成*)
2.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F] (*现在可以丢弃 W<sub>in</sub>[D, F]*)
3.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*不在关键路径上，可以在上一层期间完成*)
4.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
5.  Loss[B<sub>X</sub>] = ...

**反向传播：**需要计算 dW<sub>out</sub>[F, D<sub>X</sub>]、dW<sub>in</sub>[D<sub>X</sub>, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D<sub>X</sub>] = **ReduceScatter**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上，可以异步执行*)
4.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*可以提前完成*)
5.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D] *(此处可以丢弃 W<sub>out</sub>[F, D])*
6.  dW<sub>in</sub>[D,F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
7.  dW<sub>in</sub>[D<sub>X</sub>, F] = **ReduceScatter**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) *(不在关键路径上，可以异步执行)*
8.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*可以提前完成*)
9.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前面的层需要用到) (此处可以丢弃 W<sub>in</sub>[D, F]*)

</div>

{% enddetails %}

这也被称为“ZeRO 分片”，得名于“Zero Redundancy Optimizer”（零冗余优化器），因为我们不执行任何不必要的计算，也不存储任何不必要的状态。ZeRO-{1,2,3} 分别指以这种方式对优化器状态、梯度和权重进行分片。由于它们的通信开销都相同<d-footnote>严格来说，FSDP 在前向传播中增加了纯数据并行所没有的通信，但其比例与反向传播相同，因此不应影响通信 Roofline。这里的关键在于，ZeRO-3 把反向传播中的一次 AllReduce 变成了一次 AllGather 加一次 ReduceScatter，两者的总通信量相同。</d-footnote>，我们基本上总是可以采用 ZeRO-3 分片，即把参数、梯度和优化器状态都分片到一组设备上。

**为什么要这样做？**标准数据并行涉及大量重复工作。每个 TPU 都要对完整的梯度做 AllReduce，然后更新完整的优化器状态（所有 TPU 上的工作完全相同），再更新参数（同样完全重复）。而对于 ZeRO 分片（对梯度 / 优化器状态进行分片），你可以不做 AllReduce，而是对梯度做 ReduceScatter，只更新属于自己的那一份优化器状态分片，更新一份参数分片，然后在前向传播需要时再对参数做 AllGather。

**我们什么时候会遇到通信瓶颈？**我们的相对 FLOPs 开销和通信开销与纯数据并行完全相同，因为反向传播中的每次 AllReduce 都变成了 AllGather + ReduceScatter。回想一下，AllReduce 本身就是用一次 AllGather 和一次 ReduceScatter 实现的，两者各占一半的开销。这里我们对前向传播建模，因为它的 FLOPs 与通信之比和反向传播相同：

$$\begin{aligned}
T_\text{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_\text{comms} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}$$

因此，与纯数据并行一样，当 $$B / X > C / W_\text{ici}$$ 时我们是计算受限的，即每设备批大小 $B/X$ 超过“ICI 运算强度” $C/W_\text{ici}$（对于 v5p 为 `4.59e14 / 1.8e11 = 2550`）。这对我们来说是个好消息：这意味着，如果每设备批大小已经大到足以让纯数据并行保持计算受限，那么我们就可以直接升级到 FSDP——无需担心脱离计算受限区间——从而节省大量的参数和优化器状态内存！虽然我们确实需要在前向传播中增加通信，但这一开销无关紧要，因为它恰好可以与前向传播的 FLOPs 重叠。

<p markdown=1 class="takeaway">**要点：**在 TPUv5 上，当每设备批大小小于 $2550 / M_X$ 时（其中 $M_X$ 是网格轴的数量），FSDP 和纯数据并行都会变成带宽受限。</p>

例如，DeepSeek-V2（近期为数不多公开了训练批大小信息的强模型之一）使用了约 40M token 的批大小。**这样的批大小可以让我们扩展到大约 47,000 个芯片，也就是大约 5 个 TPUv5 Pod，才会遇到带宽限制。**

对于 LLaMA-3 70B，其训练大约用了 `6.3e24 (15e12 * 70e9 * 6)` FLOPs，我们可以把一个 16M token 的批切分到大约 `16e6 / (2550 / 3) = 18,823` 个芯片上（大约 2 个 8960 芯片的 Pod），每个芯片提供 `4.59e14` FLOPs，并以 50% 的峰值 FLOPs 利用率（通常称为 MFU）运行，这样**大约 17 天就能完成训练**。还不错！不过，我们来看看如何做得更好。

<p markdown=1 class="takeaway">**关于临界批大小的说明**：有些反直觉的是，（在芯片数量固定时）总批大小越小，我们就越容易受到通信瓶颈的限制。只要能不断增大批大小，数据并行和 FSDP 就能让我们扩展到任意多的芯片！然而在实践中，随着批大小增大，训练收益往往会递减，因为梯度几乎已经没有噪声了。有时我们还会看到训练不稳定。因此，在“无限算力情形”下寻找最优分片方案这件事，通常从一个由 scaling law 决定的固定批大小和一个已知的（很大的）芯片数量出发，目标是找到一种划分方式，使这个较小的批大小能够分布到如此多的芯片上。</p>

### 张量并行 {#tensor-parallelism}

**记法：** $$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$（我们使用 $$Y$$，是为了之后与 FSDP 结合）

在完全分片数据并行的 AllReduce 中，我们在芯片之间移动的是权重。我们也可以对模型的前馈维度进行分片，并在层的计算过程中移动激活值——这被称为“一维模型并行”或 Megatron 分片<d-cite key="megatron"></d-cite>。这可以让每个 Pod 在更小的批大小下也能高效运行。下图展示了以这种方式分片的单个矩阵的示例：

{% include figure.liquid path="assets/img/model-parallelism.png" class="img-fluid" caption="<b>图：</b>基本张量并行的示例。由于我们只在 Y 上对激活值进行分片（不同于 FSDP 中在 X 上分片），我们会在 X 上复制激活值。用我们的标准记法，这就是 <b>A</b>[B, D<sub>Y</sub>] * <b>B</b>[D, F<sub>Y</sub>] -> <b>C</b>[B, F<sub>Y</sub>]。由于我们只对其中一个收缩维度进行了分片，通常会在矩阵乘法之前对激活值 <b>A</b> 做 AllGather。" %}

如前所述，**In\[B, D<sub>Y</sub>\] \*<sub>D</sub> W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub> W<sub>out</sub>\[F<sub>Y</sub>, D\] \-\> Out\[B, D<sub>Y</sub>\] 意味着我们必须在第一次矩阵乘法之前收集（gather）激活值。当激活值比权重小时，这比 ZeRO 分片更便宜。** 通常只有在额外加入一定程度的 ZeRO 分片（这会减小 gather 的规模）时才是如此。这也是我们倾向于把 ZeRO 分片和张量并行混合使用的原因之一。

{% details 这是张量并行的算法！ %}

<div markdown=1 class="algorithm">

**张量并行：**

**前向传播：** 需要计算 Loss[B]

1.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *（位于关键路径上）*
2.  Tmp[B, F<sub>Y</sub>] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *（未沿收缩维度分片，因此无需通信）*
3.  Out[B, D] {U<sub>Y</sub>} = Tmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
4.  Out[B, D<sub>Y</sub>] = **ReduceScatter**(Out[B, D] {U<sub>Y</sub>}) *（位于关键路径上）*
5.  Loss[B] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D] 和 dW<sub>in</sub>[D, F<sub>Y</sub>]

1.  dOut[B, D<sub>Y</sub>] = ...
2.  dOut[B, D] = **AllGather**(dOut[B, D<sub>Y</sub>]) *（位于关键路径上）*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B, F<sub>Y</sub>] \*<sub>B</sub> dOut[B, D]
4.  dTmp[B, F<sub>Y</sub>] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *（此处可以丢弃 dOut[B, D]）*
5.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *（可与前向传播中的第 (1) 步共享，从而跳过这一步）*
6.  dW<sub>in</sub>[D, F<sub>Y</sub>] = In[B, D] \*<sub>B</sub> dTmp[B, F<sub>Y</sub>]
7.  dIn[B, D] {U<sub>Y</sub>} = dTmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *（前面的层需要用到）*
8.  dIn[B, D<sub>Y</sub>] = **ReduceScatter**(dIn[B, D] {U<sub>Y</sub>}) *（位于关键路径上）*

</div>

{% enddetails %}

张量并行的一个好处是，它能与 Transformer 前向传播中的两个矩阵很好地配合。朴素的做法是在两个矩阵乘法之后各做一次 AllReduce。但在这里，我们先做 **In[B, D<sub>Y</sub>] \* W<sub>in</sub>[D, F<sub>Y</sub>] -> Tmp[B, F<sub>Y</sub>]**，然后做 **Tmp[B, F<sub>Y</sub>] \* W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B, D<sub>Y</sub>]**。这意味着我们在开始时对 **In** 做 AllGather，在结束时对 **Out** 做 ReduceScatter，而不是做 AllReduce。

**这样做的开销有多大？** 我们只对前向传播建模——反向传播只是这里每个操作的转置。在一维张量并行中，我们在第一次矩阵乘法之前对激活值做 AllGather，在第二次矩阵乘法之后对其做 ReduceScatter，每个元素发送两个字节（bf16）。我们来算一算什么时候会被通信拖成瓶颈。

$$\begin{align}
T_\text{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_\text{comms} & =
\frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}$$

由于我们希望计算开销大于通信开销，可得：

$$\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}$$

$$\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}$$

$$\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}$$

因此，以 TPUv5p 为例，在 bf16 下 $C / W_{ici} = 2550$，所以张量并行最多只能做到 $Y < F / 2550$。当我们有多个 ICI 轴时，$T_\text{comms}$ 会缩小为原来的 $M_Y$ 分之一，于是得到 $Y < M_Y \cdot F / 2550$。

<p markdown=1 class="takeaway">**要点**：当 $Y > M_Y \cdot F / 2550$ 时，张量并行会变成通信受限。对大多数模型来说，这个上限在 8 路到 16 路张量并行之间。</p>

**注意，这与计算精度无关**，例如对于 int8，在 TPUv5p 上 $$C_\text{int8} / W_{ici}$$ 是 $$5100$$ 而不是 $$2550$$，但通信量也减半了，所以这两个 2 倍的因子相互抵消。

**来看几个例子：**

* 在 TPUv5p 上，对于 $$D = 8192,$$ $$F \approx 30,000$$ 的 LLaMA 3-70B，我们可以轻松地做 8 路张量并行，但做 16 路张量并行时就会通信受限。8 路模型分片所需的 F 为 20k。

* 对于 Gemma 7B，$$F \approx 50k$$，因此在 19 路张量并行时会变成通信受限。这意味着我们大概可以做 16 路张量并行，并仍然获得不错的性能。

### 结合 FSDP 与张量并行 {#combining-fsdp-and-tensor-parallelism}

**记法：** $$\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

FSDP 和张量并行的好处在于它们可以结合使用。通过沿两个轴对 **W<sub>in</sub>** 和 **W<sub>out</sub>** 进行分片，我们既节省了内存，也节省了计算。由于我们沿 X 对 B 分片，模型并行 AllGather 的规模减小了；又由于我们沿 Y 对 F 分片，FSDP 的通信开销也降低了。这意味着两者结合可以让我们达到比上面更低的有效批大小。

{% include figure.liquid path="assets/img/mixed-fsdp-model-parallelism.png" class="img-fluid" caption="<b>图：</b>结合 FSDP 与张量并行的示意图。与其他情形不同，这里模型参数没有任何重复。" %}

{% details 下面是 FSDP + 张量并行混合方案的完整算法。虽然通信很多，但由于我们对激活值做了批维度分片，并对权重做了更多的张量分片，所有 AllGather 和 ReduceScatter 的规模都更小了！ %}

<div markdown=1 class="algorithm">

**前向传播：** 需要计算 Loss[B]

1.  In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *（位于关键路径上）*
2.  W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *（可以提前完成）*
3.  Tmp[B<sub>X</sub>, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]
4.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *（可以提前完成）*
5.  Out[B<sub>X</sub>, D] {U<sub>Y</sub>} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
6.  Out[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(Out[B<sub>X</sub>, D] {U<sub>Y</sub>}) *（位于关键路径上）*
7.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] 和 dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]

1.  dOut[B<sub>X</sub>, D<sub>Y</sub>] = ...
2.  dOut[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(dOut[B<sub>X</sub>, D<sub>Y</sub>]) *（位于关键路径上）*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
4.  dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>out</sub>[F<sub>Y</sub>, D] {U<sub>X</sub>})
5.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *（可以提前完成）*
6.  dTmp[B<sub>X</sub>, F<sub>Y</sub>] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *（此处可以丢弃 dOut[B, D]）*
7. In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *（不在关键路径上，且可与上一层的第 (2) 步共享）*
8.  dW<sub>in</sub>[D, F<sub>Y</sub>] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F<sub>Y</sub>]
9.  dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>in</sub>[D, F<sub>Y</sub>] {U<sub>X</sub>})
10. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *（可以提前完成）*
11. dIn[B<sub>X</sub>, D] {U<sub>Y</sub>} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *（前面的层需要用到）*
12. dIn[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(dIn[B<sub>X</sub>, D] {U<sub>Y</sub>}) *（位于关键路径上）*

</div>

{% enddetails %}

**FSDP 和 TP 的正确组合是什么？** 一条简单但关键的准则是：FSDP 移动权重，而张量并行移动激活值。这意味着随着批大小缩小（尤其是当我们做更多数据并行时），张量并行会变得更便宜，因为每个分片上的激活值更小。

* 张量并行执行 $$\mathbf{AllGather}_Y([B_X, D_Y])$$，其规模随 $$X$$ 增大而缩小。
* FSDP 执行 $$\mathbf{AllGather}_X([D_X, F_Y])$$，其规模随 $$Y$$ 增大而缩小。

因此，将两者结合可以把每个副本的最小批大小进一步压低。我们可以用与上面相同的方法计算 FSDP 和 TP 的最优规模：

令 $$X$$ 为用于 FSDP 的芯片数，$$Y$$ 为用于张量并行的芯片数。令 $$N$$ 为切片中的芯片总数，$$N=XY$$。令 $$M_X$$ 和 $$M_Y$$ 分别为进行 FSDP 和 TP 的网格轴数量（二者之和大致应为 3）。我们只对前向传播建模，因为它每个 FLOP 对应的通信量最大。把上面算法中的通信加起来，我们有

$$T_\text{FSDP 通信}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}$$

$$T_\text{TP 通信}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}$$

同样，总的 FLOPs 时间为

$$T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.$$

为了简化分析，我们做两个假设：第一，允许 $X$ 和 $Y$ 取非整数值（只要它们为正且满足 $XY=N$）；第二，假设 $X$ 轴和 $Y$ 轴上的通信可以完全相互重叠。在第二个假设下，总通信时间为

$$T_\text{comms} = \max\left(T_\text{FSDP 通信}, T_\text{TP 通信}\right)$$

在探讨什么条件下会计算受限之前，我们先求出使总通信量最小的 $X$ 和 $Y$ 的最优值。由于 FLOPs 与 $X$ 和 $Y$ 无关，最优设置就是让通信最小的那个设置。为此，我们把上面的 $T_\text{comms}$ 用 $X$ 和 $N$（它是固定的，即系统中的芯片数）而不是 $X$ 和 $Y$ 来表示：

$$T_\text{comms} (X) = \frac{4D}{W_\text{ici}} \max\left(\frac{F \cdot X}{N \cdot M_X}, \frac{B}{X \cdot M_Y}\right)$$

由于 $T_\text{FSDP 通信}$ 关于 $X$ 单调递增，而 $T_\text{TP 通信}$ 关于 $X$ 单调递减，二者的最大值必然在 $T_\text{FSDP 通信} = T_\text{TP 通信}$ 时取到最小，此时

$$\begin{align*}
\frac{FX_{opt}}{M_X} = \frac{BN}{X_{opt} M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}$$

这非常有用！它告诉我们，在给定 $B$、$F$ 和 $N$ 时，FSDP 的最优规模是多少。我们来感受一下数量级。代入实际的数值，即 $N = 64$（对应 4x4x4 的芯片阵列）、$B=48,000$、$F=32768$，大约得到 $X\approx 13.9$。因此我们会选择 $X$ 为 16、$Y$ 为 4，接近计算出的最优值。

<p markdown=1 class="takeaway">**要点：**一般来说，在训练中，FSDP 的最优规模为 $$X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$$。</p>

现在回到我们对每种并行策略都会问的那个问题：**在什么条件下会是计算受限的？** 由于 FLOPs 和通信可以重叠，当满足以下条件时，我们就是计算受限的：

$$\max\left(T_\text{FSDP 通信}, T_\text{TP 通信}\right) < T_\text{math}$$

令 $\alpha \equiv C / W_\text{ici}$，即 ICI 算术强度，可以化简为：

$$\max\left(\frac{F}{Y \cdot M_X}, \frac{B}{X \cdot M_Y}\right) < \frac{B \cdot F}{N \cdot \alpha}$$

由于我们计算 $X_{opt}$ 时正是让左侧 max 中的两项相等，所以可以直接把它代入其中任意一项（注意 $Y_{opt} = N/X_{opt}$），即

$$\frac{F}{N \cdot W_\text{ici} \cdot M_X} \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N} < \frac{B \cdot F}{N \cdot C}$$

进一步化简，得到

$$ \sqrt{\frac{B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},$$

其中左侧正比于通信时间，右侧正比于计算时间。注意，计算时间随批大小线性增长（无论采用何种并行方式都是如此），而通信时间随批大小的平方根增长。因此，计算时间与通信时间之比也随批大小的平方根增长：

$$ \frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{\alpha \sqrt{N}}. $$

为了保证这个比值大于 1，从而是计算受限的，我们需要

$$ \frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}$$

为了得到近似数值，再次代入 $F=32,768$、$\alpha=2550$ 以及 $M_X M_Y=2$（对 3D 网格来说必然如此）。这大约给出 $B/N > 99$。与纯数据并行（或 FSDP）的情形相比，这大约为我们赢得了 8 倍的空间——在那种情形下，假设是 3D 网格，我们算出 $B/N$ 必须超过约 $850$ 才能计算受限。

<p markdown=1 class="takeaway">**要点：**将张量并行与 FSDP 结合，可以让 $B/N$ 降到 $$2550^2 / 2F$$。这让我们可以处理每个芯片低至 100 的批大小，大约比单用 FSDP 所能达到的小 8 倍。</p>

下面我们在一个有代表性的 4x4x4 芯片阵列上，绘制了 FSDP + TP 混合方案的 FLOPs 时间与通信时间之比，并与仅用张量并行（TP）和仅用数据并行（FSDP）进行比较。虽然在批大小非常大时纯 FSDP 并行占优，但在批大小与芯片数之比大约在 100 到 850 之间的区间内，必须采用 FSDP + TP 混合策略才能做到计算受限。

{% include figure.liquid path="assets/img/mixed-fsdp-comms-2.png" class="img-fluid" caption="<b>图：</b>在 F=30k 的 TPUv5p 4x4x4 切片上，最优 FSDP/TP 混合方案的 FLOPs 时间与通信时间之比。正如预期，张量并行的这一比值不随批大小变化；理想的 FSDP + TP 混合方案随 $\sqrt{B}$ 增长，而 FSDP 随 $B$ 增长。然而，在中等批大小的区间内，只有 FSDP + TP 能使这一比值大于 1。"%}

下面是 TPU v5p 16x16x16 上的另一个例子，展示了不同分片方案下 FLOPs 时间和通信时间随批大小的变化。

{% include figure.liquid path="assets/img/math-comms-time.png" class="img-fluid" caption="<b>图：</b>不同并行方案的通信耗时。黑色虚线是矩阵乘法 FLOPs 所花的时间，因此任何位于这条线之上的曲线都是通信受限的。我们注意到，所有策略在批大小低于 6e5 时都会变成通信受限，这与我们预期的 4096 * 2550^2 / (2 * 8192 * 4) = 4e5 相符。" %}

黑色曲线是花在模型 FLOPs 上的时间，这意味着在任何黑色曲线低于所有通信开销的批大小下，都严格是通信受限的。你会注意到，黑色曲线与绿色曲线大约在 `4e5` 处相交，与预测一致。

下面是一个可以动手把玩的交互式动画，展示了不同批大小下的总计算时间和通信时间：

<figure class="plotly-embed">
  <iframe src="{{ 'assets/plotly/training-roofline.html' | relative_url }}" title="混合 FSDP/TP 的 MLP Roofline" loading="lazy" scrolling="no"></iframe>
  <figcaption><b>图：</b>TPU v5p 16x16x16（$D=8192$，$F=32768$）上 MLP 前向传播的计算时间和通信时间随批大小的变化，使用的是上文的 $T_\text{FSDP 通信}$、$T_\text{TP 通信}$ 和 $T_\text{math}$。拖动滑块可以改变 FSDP/TP 的划分。阴影区域内的任何批大小都是通信受限的。</figcaption>
</figure>

你会注意到这与上面的结论基本一致：最佳划分 FSDP=512、TP=8 在批大小超过约 4e5 时变为计算受限，而任何张量并行程度更高的划分在所有批大小下都保持通信受限。张量并行程度更低的划分则需要更大的批大小。

### 流水线 {#pipelining}

你可能已经注意到，前面几节我们完全没有谈到流水线。流水线是 GPU 并行中的主流策略，但在 TPU 上则没那么必不可少。简单来说，流水线训练就是把模型的各层切分到多个设备上，并在前向和反向传播过程中在流水线阶段之间传递激活值。算法大致如下：

1. 在 TPU 0 上初始化数据，并将权重沿层维度分片（对于结合 FSDP 和张量并行的流水线，即 $W_\text{in}[L_Z, D_X, F_Y]$）。
2. 在 TPU 0 上执行第一层，然后把得到的激活值复制到 TPU 1，如此重复，直到最后一个 TPU。
3. 计算损失函数及其导数 $\partial L / \partial x_L$。
4. 对最后一个流水线阶段，计算导数 $\partial L / \partial W_L$ 和 $\partial L / \partial x_{L-1}$，然后把 $\partial L / \partial x_{L-1}$ 复制到前一个流水线阶段，如此重复，直到回到 TPU 0。

{% details 这里有一些（可运行的）Python 伪代码 %}

这段伪代码应当可以在 Cloud TPU VM 上运行。虽然它既不高效也不贴近实际，但能让你了解数据是如何在设备之间传递的。

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# 假设每一层只是一个矩阵乘法（matmul）。
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model))

def layer_fn(x, weight):
  return x @ weight

# 假设 num_layers == num_pipeline_stages
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # 随便编一个假的损失函数

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(num_layers - 1, -1, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i], weights[i])
  dx, dw = f_vjp(dx)  # 计算 jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # 更新我们的权重

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

{% enddetails %}

**为什么这是个好主意？** 流水线有很多优点：流水线阶段之间的通信开销很低，这意味着即使互连带宽很低，也能训练非常大的模型。这在 GPU 上往往非常有用，因为 GPU 不像 TPU 那样通过 ICI 密集互连。

**为什么这很难/很烦人？** 你可能已经在上面的伪代码中注意到，TPU 0 几乎一直处于空闲状态！它只在流水线的第一步和最后一步做工作。这段空闲时间称为流水线气泡（pipeline bubble），处理起来非常麻烦。通常我们首先尝试用微批次来缓解它，即把多个小批次依次送入流水线，使 TPU 0 至少在总步时间中更大的一部分里保持忙碌。

第二种方法是仔细地重叠前向矩阵乘法 $W_i @ x_i$、反向的 $dx$ 矩阵乘法 $W_i @ \partial L / \partial x_{i+1}$ 以及 $dW$ 矩阵乘法 $\partial L / \partial x_{i+1} @ x_i$。由于每一项都需要一些 FLOPs，我们可以把它们重叠起来，从而完全隐藏气泡。下面是最近的 DeepSeek v3 论文<d-cite key="DeepSeek3"></d-cite>中的一张图，展示了他们“无气泡”的流水线调度：

{% include figure.liquid path="assets/img/deepseek-pipeline.png" class="img-fluid" caption="<b>图：</b>DeepSeek v3 的流水线调度（来自他们的<a href=\"https://arxiv.org/pdf/2412.19437\">最新论文</a>）。橙色是前向矩阵乘法，绿色是 dL/dx 矩阵乘法，蓝色是 dL/dW 矩阵乘法。通过优先执行反向的 dL/dx 乘法，我们可以避免 FLOPs 被“搁浅”。" %}

由于流水线对 TPU 来说没那么关键（TPU 拥有更大的互连 Pod），我们不会深入探讨它，但理解流水线的关键瓶颈是一个很好的练习。

### 跨 Pod 扩展 {#scaling-across-pods}

最大的 TPU 切片是拥有 8960 个芯片（以及 2240 台主机）的 TPU v5p SuperPod。当我们想扩展到超过这个规模时，就需要跨越数据中心网络（Data-Center Networking，DCN）的边界。每台 TPU 主机都配备一个或多个 NIC（网络接口卡），通过以太网把主机连接到其他 TPU v5p Pod。正如 [TPU 章节](../tpus)中所述，每台主机大约有 200Gbps（25GB/s）的全双工 DCN 带宽，相当于每个 TPU 约 6.25GB/s 的全双工（出口）带宽。

通常，当扩展到单个 Pod 之外时，我们会在 ICI 域内做某种形式的模型并行或 FSDP，然后在多个 Pod 之间做纯数据并行。令 $N$ 为我们想要扩展到的 TPU 数量，$M$ 为每个 ICI 互连切片中的 TPU 数量。为了在 DCN 上做 AllReduce，我们可以在这组 Pod 上做环形归约（ring-reduction），从而（在反向传播中）得到：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{M \cdot W_\text{dcn}}$$

通信带宽随 $M$ 增长，因为与 ICI 不同，随着 ICI 域变大、拥有的 NIC 变多，总带宽也会增长。化简后可以发现，$T_\text{math} > T_\text{comms}$ 的条件是

$$\frac{B}{\text{切片}} > \frac{C}{W_\text{dcn}}$$

对于 TPU v5p，$\frac{C}{W_\text{dcn}}$ 约为 `4.59e14 / 6.25e9 = 73,440`。这告诉我们，要在 DCN 上高效扩展，每个 ICI 域都需要达到一个最小批大小，才能撑得住每个节点的出口通信。

**这个问题有多严重？** 举个具体的例子，假设我们想在 TPU v5p 上以 2M token 的批大小（BS）训练 LLaMA-3 70B。LLaMA-3 70B 的 $F\approx 30,000$。根据前面几节，我们知道：

* 张量并行最多可以做到 $Y = M_Y \cdot F / 2550 \approx 11 \cdot M_Y$。
* 只要 $B / N > 2550 / M_X$，我们就可以做 FSDP。这意味着如果想以 BS=2M、3 个数据并行轴进行训练，最多只能使用 $\approx 2400$ 个芯片，大约是一个 TPU v5p Pod 的四分之一。
* 当我们结合 FSDP + 张量并行时，在 $B / N < 2550^2 / (2 \cdot 30000) = 108$ 时才会变成通信受限，因此这让我们可以扩展到大约 18k 个芯片！然而，TPU v5p Pod 的最大规模是 8k 个芯片，超出这个规模就必须使用 DCN。

简而言之（TLDR），我们有一个很好的方案来以 BS=1M 进行训练，大约使用 X（FSDP）= 1024 和 Y（TP）= 8，但对于 BS=2M 就需要使用 DCN。如上所述，DCN 算术强度为 $\text{73,440}$，所以我们只需确保每个 ICI 域的批大小大于这个值即可。这对我们来说轻而易举，因为用 2 个 Pod 时，每个 Pod 的 BS 为 1M，每个 TPU 的批大小为 111，这非常好（也许有点紧，但理论上是可行的）。

<p markdown=1 class="takeaway">**要点：**只要每个 Pod 的批大小至少为 73k token，使用纯数据并行跨多个 TPU Pod 扩展就相当直接。</p>

## 在 TPU 上训练 LLM 的要点 {#takeaways-from-llm-training-on-tpus}

* 增加并行度或减小批大小都往往会让我们更趋于通信受限，因为二者都会减少每个芯片上执行的计算量。

* 在合理的上下文长度（约 32k）以内，我们可以把 Transformer 近似建模为一摞 MLP 块，并根据每层两个/三个主要矩阵乘法的分片方式来定义各种并行方案。

* 训练时我们主要考虑 4 种并行方案，每种都有自己的带宽和计算需求（数据并行、FSDP、张量并行，以及 FSDP + 张量并行的混合方案）。

| **策略**                                 | **描述**                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **数据并行**                         | 激活值按批次分片，其余一切都完全复制，在反向传播期间对梯度做 AllReduce。                                                                      |
| **FSDP**                                     | 激活值、权重和优化器状态都按批次分片，权重在使用前才被 gather，梯度做 ReduceScatter。                                                               |
| **张量并行（又称 Megatron 并行、模型并行）** | 激活值沿 $$d_\text{model}$$ 分片，权重沿 $$d_{ff}$$ 分片，激活值在 W<sub>in</sub> 之前做 gather，结果在 W<sub>out</sub> 之后做 ReduceScatter。 |
| **FSDP + 张量并行混合**          | 同时使用以上两者，其中 FSDP gather 的是已按模型并行（张量并行）分片的权重。                                                                                                                           |

下面是每种方法的“公式”：

$$\small
\begin{array}{cc}
\text{策略} & \text{公式}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{TP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{TP + FSDP}  & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}$$

* 根据每个设备上的计算量和通信量，每种策略都有一个界限，超过它就会变成网络/通信受限。下面是每层的计算量和通信量，假设 $$X$$ 是 FSDP 轴、$$Y$$ 是张量并行轴。

$$
\small
\begin{array}{ccc}
\text{策略} & \text{每层计算} & \text{每层通信} \\
& \text{（忽略门控 einsum）} & \text{（字节，前向 + 反向传播）}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{TP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + TP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}$$

* 纯数据并行很少有用，因为模型及其优化器状态占用的字节数 = 参数量的 10 倍。这意味着我们很少能在内存中放下超过几十亿个参数。

* 当 $$\text{每分片批大小} < C / W$$（即网络的算术强度）时，数据并行和 FSDP 就会变成通信受限。对 ICI 来说这个值是 2,550，对 DCN 来说约为 71,000。使用更多并行轴可以提高这个值。

* 当 $$\lvert Y\rvert > F / 2550$$ 时，张量并行会变成通信受限。**对大多数模型来说，这大约是 8-16 路。** 这与批大小无关。

* FSDP + 张量并行混合方案让我们能把批大小降到低至 $$2550^2 / 2F \approx 100$$。这已经非常低了。

* 跨 Pod 的数据并行要求每个 Pod 的批大小至少约为 71,000，否则就会变成 DCN 受限。

* 总的来说，如果批大小很大或模型很小，事情就很简单：你可以用数据并行，或者在 DCN 上用 FSDP + 数据并行。真正有意思的是中间地带。

## 一些练习题 {#some-problems-to-work}

本节我们用 LLaMA-2 13B 作为基础模型。模型细节如下：

| 超参数 | 取值  |
| ---------- | ------ |
| L          | 40     |
| D          | 5,120  |
| F          | 13824  |
| N          | 40     |
| K          | 40     |
| H          | 128    |
| V          | 32,000 |

LLaMA-2 的嵌入矩阵和输出矩阵是分开的，并且使用门控 MLP 块。

**问题 1：** LLaMA-2 13B 有多少参数（我知道这问题有点傻，但还是算一下吧）？*注意，和 [Transformer 数学](../transformers)中一样，LLaMA-3 有 3 个大的 FFW 矩阵：两个上投影矩阵和一个下投影矩阵。本节中我们忽略了两个“门控”einsum 矩阵，但它们的行为与本节中的 W<sub>in</sub> 相同。*

{% details 点击此处查看答案。 %}

* FFW 参数：$$3LDF$$ = `8.5e9`
* 注意力参数：$$4DNHL$$ = `4.2e9`
* 词表参数：$$2VD$$ = `0.33e9`
* 总计：`8.5e9 + 4.2e9 + 0.33e9 = 13.0e9`，与预期一致！

{% enddetails %}

**问题 2：** 假设我们以 BS=16M token 进行训练，并使用 Adam。暂时忽略并行，模型的参数、优化器状态和激活值一共占用多少内存？*假设参数以 bf16 存储、优化器状态以 fp32 存储，并且每层对激活值做三次检查点（在三个大的 FFW 矩阵乘法之后）。*

{% details 点击此处查看答案。 %}

参数（bf16）和两个优化器状态（fp32，即一阶矩和二阶矩累加器）总共占用的内存为 `(2 + 4 + 4) * 13e9 ~ 130GB`。前两个矩阵乘法之后的激活值形状为 $BF$，最后一个之后为 $BD$（见上面的 Transformer 示意图），因此以 bf16 存储时总内存为 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$，即 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`，因为 `B=16e6`。其他激活值基本都可以忽略。

{% enddetails %}

**问题 3：** 假设我们想在一个 TPUv5p 16x16x16 切片上，以 32k 序列长度、总批大小 3M token 进行训练。和上面一样，假设我们使用 bfloat16 权重和 float32 优化器。

1. 我们能用纯数据并行吗？为什么能或为什么不能？
2. 我们能用纯 FSDP 而不陷入通信受限吗？为什么能或为什么不能？使用纯 FSDP 时，每个设备会占用多少内存（假设只在 3 个大的 FFW 矩阵之后做梯度检查点）？
3. 我们能用 FSDP + 张量并行混合方案吗？为什么能或为什么不能？如果能，$X$ 和 $Y$ 应该取多少？每个设备会存储多少内存？只用 Roofline FLOPs 估算并忽略注意力，在 40% MFU 下每个训练步需要多长时间？

{% details 点击此处查看答案。 %}

首先写下一些数字。32k 序列长度、3M 批大小，意味着以序列计的批大小为 96。在一个 TPU v5p 16x16x16 切片上，我们共有 `393TB` 的 HBM。

1. 我们不能用纯数据并行，因为它会在每个芯片上复制参数和优化器状态，这些已经约有 130GB（来自问题 2），比每个芯片的 HBM（96GB）还多。

2. 先只看内存。把问题 2 中的 BS=16M 换成 3M，我们得到 `~7.86e12` 的检查点激活值总量，再加上 1.3e11 的优化器状态，几乎正好是 8e12 = 8TB。TPUv5p 切片总共有 `393TB` 的 HBM，所以我们稳稳处在 HBM 上限之内。接下来看我们是通信受限还是计算受限。用 4096 个芯片和 3 个并行轴，我们能做到的最小批大小为 `850 * 4096 = 3.48M` token。这比我们 3M 的批大小略高。所以我们实际上是通信受限的，这很遗憾。因此总体答案是：**不能，单独使用 FSDP 无法避免通信受限**。

3. 现在我们知道主要的顾虑是通信受限，那就代入一些数字。首先，由上文可知，使用 FSDP + 张量并行混合方案时，每个芯片的批大小需要高于 $2550^2 / 2F = 235$。这意味着理论上我们可以做到！我们来算一下两者各取多少。

我们有规则 $X_{opt} = \sqrt{(B / F) \cdot (M_X / M_Y) \cdot N}$，这里得到 `sqrt(3e6 * 2 * 4096 / 13824) = 1333`，也就是说我们大约做 1024 路 DP 和 4 路 TP。每个 TPU 的内存与 (2) 中相同，而步时间就是 `6 * 3e6 * 13e9 / (4096 * 4.6e14 * 0.4) = 300ms`。

{% enddetails %}

<h3 markdown=1 class="next-section">第 5 部分到此结束！第 6 部分会把这些内容应用到真实的 LLaMA 模型上，[点击这里](../applied-training)继续！</h3>

## 附录 {#appendix}

### 附录 A：推导反向传播的通信 {#appendix-a-deriving-the-backward-pass-comms}

上文中，我们把 Transformer 层的前向传播简化为 Out[B, D] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F] *<sub>F</sub> W<sub>out</sub>[F, D]。那么反向传播所需的通信该如何推导呢？

这可以很自然地从上一节中关于单个矩阵乘法 **Y = X * A** 的规则得出：

$$\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)$$

$$\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T$$

据此，我们得到以下公式（令 Tmp[B, F] 表示 In[B, D] * W<sub>in</sub>[D, F]）：

<div markdown=1 class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
2. dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
3. dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
4. dIn[B, D] = dTmp[B, F] *<sub>F</sub> W<sub>in</sub>[D, F]

</div>

注意，这些公式只是数学表述，并未涉及分片。反向传播的任务就是计算这四个量。因此，要弄清所需的通信，我们只需取上面四个等式中所有参与矩阵乘法的量（Tmp、dOut、W<sub>out</sub>、W<sub>in</sub>）的分片方式——它们由我们的并行方案决定——然后用分片矩阵乘法的规则来确定需要做哪些通信。注意，dOut 的分片方式与 Out 相同。
