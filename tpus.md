---
layout: distill
title: "如何理解 TPU"
# permalink: /main/
description: "本节全面介绍 TPU 的工作原理、TPU 如何互连以支持多芯片训练和推理，以及这些因素如何影响我们最喜爱的算法的性能。其中也有不少对 GPU 用户有用的内容！"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting

section_number: 2

previous_section_url: "../roofline"
previous_section_name: "第 1 部分：Roofline 模型"

next_section_url: ../sharding
next_section_name: "第 3 部分：分片"

bibliography: main.bib

giscus_comments: false
giscus_discussion: 4

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
  - name: 什么是 TPU？
    id: what-is-a-tpu
  - name: TPU 网络
    id: tpu-networking
  - name: 关键要点
    id: key-takeaways
  - subsections:
    - name: TPU 规格
      id: tpu-specs
  - name: 练习题
    id: worked-problems
  - name: 附录
    id: appendix
  - subsections:
    - name: "附录 A：TPU 内部结构详解"
      id: appendix-a-more-on-tpu-internals
    - name: "附录 B：脉动阵列是如何工作的？"
      id: appendix-b-how-does-a-systolic-array-work

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

<p markdown=1 class="announce">你可能也会喜欢阅读关于 NVIDIA GPU 的新章节 [第 12 部分](../gpus)！</p>

## 什么是 TPU？ {#what-is-a-tpu}

**TPU 本质上就是一个专门做矩阵乘法的计算核心（称为 TensorCore），连接着一叠高速内存（称为高带宽内存，即 HBM）<d-cite key="tpu_paper"></d-cite>。** 下面是一张示意图：

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>图：</b>TPU 芯片的基本组成部分。TensorCore 是左侧的灰色方框，其中包含矩阵乘法单元（MXU）、向量单元（VPU）和向量内存（VMEM）。" %}

你可以把 TensorCore 基本上看作一台非常出色的矩阵乘法机器，不过它还有其他几项值得注意的功能。TensorCore 有三个关键单元：

* **MXU**（Matrix Multiply Unit，矩阵乘法单元）是 TensorCore 的核心。在大多数 TPU 代际中，它利用脉动阵列（systolic array）每 8 个周期完成一次 `bf16[8,128] @ bf16[128,128] -> f32[8,128]` 矩阵乘法<d-footnote>TPU v6e（Trillium）的 MXU 为 256x256，而之前所有代际都是 128x128。</d-footnote>（详见<a href="#appendix-b-how-does-a-systolic-array-work">附录 B</a>）。
  * 在 TPU v5e 上，以 1.5GHz 计算，每个 MXU 大约能提供 `5e13` bf16 FLOPs/s。大多数 TensorCore 有 2 个或 4 个 MXU，因此例如 TPU v5e 的 bf16 总 FLOPs/s 为 `2e14`。
  * TPU 也支持吞吐量更高的低精度矩阵乘法（例如每块 TPU v5e 芯片可以达到 `4e14` int8 OPs/s）。

* **VPU**（Vector Processing Unit，向量处理单元）执行通用数学运算，比如 ReLU 激活，或者向量之间的逐点加法、乘法。归约（求和）也在这里完成。<a href="#appendix-a-more-on-tpu-internals">附录 A</a> 提供了更多细节。
* **VMEM**（Vector Memory，向量内存）是位于 TensorCore 内部、紧邻计算单元的片上暂存存储器（scratchpad）。它比 HBM 小得多（例如在 TPU v5e 上为 128 MiB），但到 MXU 的带宽要高得多。VMEM 的作用有点像 CPU 上的 L1/L2 缓存，但容量大得多，而且由程序员控制。HBM 中的数据需要先复制到 VMEM，TensorCore 才能用它进行计算。

**TPU 做矩阵乘法非常非常快**。这基本上就是它们的主要工作，而且做得很好。[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 是迄今最强大的 TPU 之一，每个核心每秒可完成 `2.5e14` bf16 FLOPs，每块芯片每秒可完成 `5e14` bf16 FLOPs。一个由 8960 块芯片组成的 Pod 可以达到 4 bf16 exaFLOPs/s。这是*非常多*的算力，相当于世界上最强大的超级计算机之一。而 Google 拥有很多这样的 Pod。<d-footnote>TPU，尤其是其中的脉动阵列，之所以是如此强大的硬件加速器，是因为矩阵乘法是少数几个对 $O(n^2)$ 字节数据执行 $O(n^3)$ 计算量的算法之一。这使得普通的 ALU 很容易受限于计算，而不是受限于内存带宽。</d-footnote>

上图中还包含 SMEM 和标量单元等其他几个组件，它们用于处理控制流，在<a href="#appendix-a-more-on-tpu-internals">附录 A</a> 中有简要讨论，但理解它们并不关键。相比之下，HBM 很重要，而且相当简单：

* **HBM**（High Bandwidth Memory，高带宽内存）是一大块高速内存，用于存放供 TensorCore 使用的张量。HBM 的容量通常在几十 GB 的量级（例如，[TPU v5e 拥有 16GiB 的 HBM](https://cloud.google.com/tpu/docs/v5e#system_architecture)）。

  * 当计算需要时，张量会从 HBM 经由 VMEM（见下文）流式传入 MXU，计算结果再从 VMEM 写回 HBM。

  * HBM 与 TensorCore 之间（经由 VMEM）的带宽称为“HBM 带宽”（通常约为 1-2TB/sec），它限制了内存受限工作负载的计算速度。

**一般来说，TPU 的所有操作都是流水线化并相互重叠的。** 要执行矩阵乘法 $X \cdot A \to Y$，TPU 首先需要把矩阵 $A$ 和 $X$ 的分块从 HBM 复制到 VMEM，然后将其载入 MXU，由 MXU 对 8x128（$X$ 的分块）和 128x128（$A$ 的分块）的块做乘法，再把结果逐块复制回 HBM。为了高效完成这一过程，矩阵乘法采用流水线方式执行，使进出 VMEM 的复制与 MXU 的计算相互重叠。这样 MXU 就能持续工作而不必等待内存传输，从而让矩阵乘法保持计算受限，而不是内存受限。

下面是一个从 HBM 读取数据并执行逐元素乘积的例子：

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>图：</b>动画展示了在 TPU 上执行的逐点乘积，字节从 HBM 中载入。注意字节是如何分块从内存中流式读出的，部分结果也以流水线方式写回，而无需等待整个数组完全物化。" %}

矩阵乘法的过程看起来几乎完全一样，只是数据会载入 MXU 而不是 VPU/向量单元，并且载入和存储的顺序有所不同，因为同一个权重分块会被用于多个激活值分块。你可以看到数据块先流入 VMEM，再进入 VREGs（向量寄存器），然后进入向量单元，最后回到 VMEM 和 HBM。我们马上就会看到，如果从 HBM 到 VMEM 的载入比向量单元（或 MXU）中的 FLOPs 更慢，我们就会变成“带宽受限”，因为 VPU 或 MXU 没有足够的工作可做。

<p markdown=1 class="takeaway">**关键要点：** TPU 非常简单。它们把权重从 HBM 载入 VMEM，再从 VMEM 载入一个每秒可执行约 200 万亿次乘加运算的脉动阵列。HBM $\leftrightarrow$ VMEM 以及 VMEM $\leftrightarrow$ 脉动阵列之间的带宽，从根本上限制了 TPU 能够高效完成哪些计算。</p>

**VMEM 与算术强度：** VMEM 比 HBM 小得多，但到 MXU 的带宽要高得多。正如我们在[第 1 部分](../roofline)中看到的，这意味着如果一个算法能把所有输入/输出都放进 VMEM，它就不太容易遇到通信瓶颈。当计算的算术强度较低时，这一点尤其有帮助：VMEM 带宽大约是 HBM 带宽的 22 倍，这意味着一个从 VMEM 读取/向 VMEM 写入的 MXU 操作只需要 10-20 的算术强度就能达到峰值 FLOPs 利用率。也就是说，如果我们能把权重放进 VMEM 而不是 HBM，矩阵乘法在小得多的批大小下就能达到 FLOPs 受限。这也意味着那些本质上算术强度较低的算法依然可以高效运行。只是 VMEM 实在太小，这往往是个挑战。<d-footnote>我们有时会谈到 VMEM 预取（prefetching），指的是提前把权重载入 VMEM，从而掩盖矩阵乘法的载入开销。例如，在普通的 Transformer 中，我们有时可以在计算注意力的同时把大的前馈层权重载入 VMEM；如果我们是内存带宽受限的，这样就能隐藏权重载入的开销。这要求权重足够小或分片得足够细，使单层权重能够放进 VMEM 并留有余量。</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**一块 TPU 芯片通常（但并非总是）由两个共享内存的 TPU 核心组成，可以看作一个拥有两倍 FLOPs 的大型加速器**（称为“megacore”配置）。v4、v5 和 v6 TPU 都是如此（TPU v7 取消了 megacore，改为在两个核心之间提供高带宽链路）。较老的 TPU 芯片（TPU v3 及更早）的两个核心各有独立内存，被视为两个独立的加速器。像 TPU v5e 这样针对推理优化的芯片，每块芯片只有一个 TPU 核心。

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**芯片**以 **4 块为一组安装在一个“托盘”（tray）上**，并**通过 PCIe 网络连接到一台 CPU 主机。** 这是大多数读者熟悉的形式：通过 Colab 或单个 TPU-VM 暴露出来的 4 块芯片（8 个核心，但通常被视为 4 个逻辑 megacore）。对于 TPU v5e 这样的推理芯片，每台主机有 2 个托盘而不是 1 个，但每块芯片也只有 1 个核心，因此共有 8 块芯片 = 8 个核心。<d-footnote>在 Cloud TPU VM 上，每个托盘作为一个单独 VM 的一部分暴露出来，因此可见的依然是 4 个核心。</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe 带宽有限：** 与 HBM $\leftrightarrow$ VMEM 链路一样，CPU $\leftrightarrow$ HBM 的 PCIe 连接也有特定的带宽，它限制了从主机内存载入数据到 HBM（或反向传输）的速度。例如，TPU v4 的 PCIe 带宽为每个方向 16GB / second，比 HBM 慢了将近 100 倍。我们*可以*把数据载入/卸载到主机（CPU）内存中，但速度不会很快。

## TPU 网络 {#tpu-networking}

**在一个 Pod 内，芯片之间通过 ICI 网络相互连接**。在较老的代际（TPU v2 和 TPU v3）、推理芯片（例如 TPU v5e）以及 Trillium（TPU v6e）中，ICI（“inter-chip interconnects”，芯片间互连）连接最近的 4 个邻居（并通过边缘链路形成二维环面）。TPU v4 和 TPU v5p 则连接最近的 6 个邻居（形成三维环面）。注意，这些连接**不**经过它们的主机，而是芯片之间的直接链路。

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

环面（torus）结构把任意两个节点之间的最大距离从 $N$ 降低到 $N / 2$，使通信快得多。TPU 还有一种“扭曲环面”（twisted torus）配置，它把环面以类似莫比乌斯带的拓扑方式环绕起来，以进一步缩短节点之间的平均距离。

**TPU Pod（通过 ICI 连接）可以变得非常大：** 最大的 Pod 规模（称为 **SuperPod**）对于 TPU v4 是 `16x16x16`，对于 TPU v5p 是 `16x20x28`。这些大型 Pod 由可重配置的 `4x4x4` 芯片立方体组成，立方体之间通过[光学环绕链路](https://arxiv.org/pdf/2208.10041)相连<d-footnote>光交换机只是一种具有相同 ICI 带宽的可重配置连接。它让我们能够在连接多个立方体的同时保留环绕链路。</d-footnote>，我们可以通过重新配置这些链路来连接非常大的拓扑。

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

也可以申请更小的拓扑（例如 `2x2x1`、`2x2x2`），但它们没有环绕连接。这是一个重要的注意事项，因为这通常会使大多数通信的耗时翻倍。完整立方体的任意整数倍（例如 `4x4x4` 或 `4x4x8`）都会有由光交换机提供的环绕连接。<d-footnote>注意，`2x2x4` 不会有任何环绕连接，因为环绕连接由光交换机提供，而光交换机只在完整立方体上可用。不过，TPU v5e 8x16 <em>会</em>在较长的轴上有环绕连接，因为它不使用可重配置的光网络。</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e 和 Trillium 的 Pod 由单个 `16x16` 二维环面构成，任何大小为 16 的轴上都有环绕连接（这意味着 `8x16` 在长轴上有环绕连接）。TPU v5e 和 v6e（Trillium）无法扩展到 16x16 环面之外，但 Pod 之间仍然可以通过标准的数据中心网络（DCN）通信，DCN 负责把 TPU 主机相互连接起来。同样，也可以申请更小的拓扑，但在大小 $<16$ 的维度上没有环绕连接。

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**这种最近邻连接方式是 TPU 与 GPU 之间的一个关键区别**。GPU 通过多层级的交换机相连，近似实现每对 GPU 之间的点对点连接，而不是像 TPU 那样使用局部连接。通常，一个节点内的 GPU（H100 为 8 块 GPU，B200 NVL72 则多达 72 块）是直接相连的，而更大的拓扑则需要在每对 GPU 之间经过 O(log(N)) 跳。一方面，这意味着 GPU 可以在少量跳数内发送任意数据。另一方面，TPU 要便宜得多（因为 NVLink 交换机很贵），布线也更简单，并且能够扩展到大得多的拓扑，因为每个设备的链路数和每个设备的带宽都是恒定的。更多内容请阅读[这里](../gpus#networking)。

**ICI 相对于 DCN 非常快，但仍然比 HBM 带宽慢。** 例如，[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 具有：

* 每块芯片 `2.8e12` bytes/s（2.8 TB/s）的 HBM 带宽。
* 每个轴 `9e10` bytes/s（90 GB/s）的 ICI 带宽，每块芯片有 3 个轴。<d-footnote>上面链接的页面列出的带宽是 100 GB/s，与这里列出的略有不同。TPU ICI 链路的带宽会因所执行的操作不同而略有差异。一般来说，你可以放心使用本文中的数字。</d-footnote>
* 每个 TPU `6.25e9` bytes/s（6.25 GB/s）的 DCN（出口）带宽（通过每台主机上的 1-2 块网卡）。<d-footnote>TPU v6e 和 TPU7x 为 12.5e9 bytes/s，v5e 为 3.125e9 bytes/s。</d-footnote>

这意味着，当我们把模型拆分到多块芯片上时，需要小心避免较慢的跨设备通信成为 MXU 的瓶颈。

**多切片训练：** 一组通过 ICI 相连的 TPU 称为一个**切片（slice）**。不同的切片之间可以通过 DCN 相连，例如用来连接位于不同 Pod 上的切片。由于 DCN 比 ICI 慢得多，我们应该尽量减少计算等待 DCN 数据的时间。DCN 是主机到主机的连接，因此要通过 DCN 在 TPU 之间传输缓冲区，我们首先需要通过 PCIe 传输到主机，然后从网络出口发出，再经目标主机的网络入口接收，最后通过 PCIe 传入 HBM。

## 关键要点 {#key-takeaways}

* TPU 很简单，在大多数情况下可以把它看作一个矩阵乘法单元，连接着内存（超快）、通过 ICI 连接的其他芯片（相当快），以及通过 DCN 连接的数据中心其余部分（还算快）。

* 通信受到各种网络带宽的限制，按速度从快到慢排列如下：
  * HBM 带宽：TensorCore 与其对应的 HBM 之间。
  * ICI 带宽：TPU 芯片与其最近的 4 个或 6 个邻居之间。
  * PCIe 带宽：CPU 主机与其对应的芯片托盘之间。
  * DCN 带宽：多台 CPU 主机之间，通常是未通过 ICI 相连的主机之间。

* **在一个切片内，TPU 只通过 ICI 与其最近的邻居相连。** 这意味着切片中相距较远的芯片之间通过 ICI 通信时，需要先跳经中间的芯片。

* **权重矩阵的两个维度都需要填充（pad）到至少 128**（在 TPU v6e 上为 256）才能填满 MXU（事实上，较小的轴会被填充到 128）。

* **较低精度的矩阵乘法往往更快。** 对于支持的代际，TPU 执行 int8 或 int4 OPs 的速度大约是 bfloat16 FLOPs 的 2 倍/4 倍。VPU 操作仍以 fp32 执行。

* 为了避免 TPU 计算单元出现瓶颈，我们需要**确保每个通道上的通信量与其速度成正比**。

### TPU 规格 {#tpu-specs}

下面是我们这些芯片的一些具体数字：

| 型号                                       | Pod 规模 | 主机规模 | 每芯片 HBM 容量 | 每芯片 HBM 带宽（bytes/s） | 每芯片 FLOPs/s（bf16） | 每芯片 FLOPs/s（int8） |
| :----------------------------------------- | :------: | :-------: | :---------------: | :-------------------: | :-----------------: | :-----------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |       32GB        |        9.0e11         |       1.4e14        |       1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |       32GB        |        1.2e12         |       2.75e14       |       2.75e14       |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |       96GB        |        2.8e12         |       4.59e14       |       9.18e14       |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |       16GB        |        8.2e11         |       1.97e14       |       3.94e14       |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |       32GB        |        1.6e12         |       9.20e14       |       1.84e15       |
| <span class="nowrap-header">TPU7x</span>   | 4x4x576  |   2x2x1   |       192GB       |        7.4e12         |       2.30e15       |       4.61e15       |

主机规模指的是连接到单台主机的 TPU 的拓扑（例如，TPU v5e 的单台 CPU 主机以 4x2 拓扑连接 8 块 TPU）。关于最新一代的更多细节，请参阅 [TPU7x 文档](https://docs.cloud.google.com/tpu/docs/tpu7x)。下面是互连数据：

| 型号        | 每链路 ICI 带宽（单向，bytes/s） | 每链路 ICI 带宽（双向，bytes/s） |
| :---------- | :----------------------------: | :-------------------------: |
| **TPU v3**  |             1.0e11             |           2.0e11            |
| **TPU v4p** |             4.5e10             |           9.0e10            |
| **TPU v5p** |             9.0e10             |           1.8e11            |
| **TPU v5e** |             4.5e10             |           9.0e10            |
| **TPU v6e** |             9.0e10             |           1.8e11            |
| **TPU7x**   |             9.0e10             |           1.8e11            |

我们同时列出了单向（unidirectional）带宽和双向（bidi，bidirectional）带宽，因为单向带宽更贴近硬件的真实情况，而双向带宽在涉及完整环的公式中出现得更频繁。<d-footnote>所谓双向（bidirectional）带宽，指的是单条链路在两个方向上总共能发送的字节数，或者等价地说，假设我们能高效利用两条链路，单个 TPU 沿某个特定轴发出的总字节数。当我们拥有一个完整可用的环时，也就是在该轴上有环绕连接时，这一点成立。对于推理芯片，这发生在轴大小为完整的 16 时；对于训练芯片（v*p），则发生在轴大小为 4 的倍数时。我们倾向于使用双向带宽，因为它在涉及双向通信的计算中经常出现。</d-footnote>

PCIe 带宽通常约为每个 TPU `1.6e10` bytes / second（TPU v6e 为 `3.2e10`），而 DCN 带宽通常约为每个 TPU `6.25e9` bytes / second（TPU v6e 和 TPU7x 为 `12.5e9`，TPU v5e 为 `3.125e9`）。

## 练习题 {#worked-problems}

这些数字有点枯燥，但有了它们，你就能对模型性能做基本的 Roofline 估算。下面我们做几道题来说明它们为什么有用。在第 3 部分中你会看到更多例子。

**问题 1 [LLM 延迟的下界]：** 假设你想从一个 200B 参数、bf16 精度、切分到 32 块 TPU v4p 上的模型中采样。把所有参数从 HBM 加载到脉动阵列中需要多长时间？*提示：使用上面的数字。*

{% details 点击此处查看答案。 %}

**答案：** 我们要在 32 块芯片上加载 `sizeof(bf16) * 200e9 = 400e9` 字节，即每块芯片 12.5e9 字节，每块芯片的 HBM 带宽为 1.23e12。所以加载大约需要 10ms。

这很有意思，因为*这是从该模型采样的延迟的一个合理下界*。每个采样步骤都需要从 HBM 加载全部参数，所以不可能少于 10 ms。实际上，在小批大小下，这个下界几乎是可以达到的。

{% enddetails %}

**问题 2 [TPU 细节]：** 考虑一个完整的 TPU v5e Pod。总共有多少个 CPU 主机？多少个 TPU TensorCore？整个 Pod 的总 FLOPs/s 是多少？总 HBM 是多少？对 TPU v5p Pod 做同样的练习。

{% details 点击此处查看答案。 %}

**答案：** 对于 TPU v5e，每个 Pod 是 `16x16`，每个主机是一个 4x2 的切片，所以共有 `16*16 / 8 = 32` 个主机。对于 TPU v5e，每块 TPU 只有一个核心，所以共有 256 个 TensorCore。bfloat16 下的总 FLOPs/s 为 `16*16*2e14 = 5.1e16`。每块芯片有 16GB HBM，所以总内存为 `256 * 16 = 4TB`。

对于一个完整的 TPU v5p Pod，共有 `16x20x28` 块芯片，每个主机是 2x2x1，所以共有 `(16*20*28) / (2*2) = 2,240` 个主机。对于 TPU v5p，每块 TPU 有两个 TensorCore，所以共有 `8960 * 2 = 17,920` 个核心。bfloat16 下的总 FLOPs/s 为 `8960 * 4.59e14 = 4.1e18`。每块芯片有 96GB HBM，所以总内存为 `8960 * 96 = 860TB`。

{% enddetails %}

**问题 3 [PCIe 运算强度]：** 设想我们被迫把一个类型为 $\text{bf16}[D, F]$ 的大权重矩阵 $A$ 和一批类型为 $\text{bf16}[B, D]$ 的激活值 $x$ 存放在主机 DRAM 中，并想对它们做矩阵乘法。程序运行在单个主机上，我们使用挂在该主机上的一块 TPU v6e 芯片。你可以假设 $B \ll D$，且 $F = 4D$（我们会在后续章节中看到为什么这些假设是合理的）。为了在 PCIe 上保持 FLOPs 受限，我们所需的最小批大小 $B$ 是多少？假设 PCIe 带宽为 1.6e10 字节/秒。

{% details 点击此处查看答案。 %}

**答案：** 我们需要执行 $2BDF$ 次浮点运算，每块芯片每秒可以执行 `9.2e14` 次浮点运算。因此计算需要 $2BDF / 9.2e14$ 秒。我们需要从 DRAM 加载 $2DF + 2BD$ 字节，并向其写回 $2BF$ 字节。瓶颈在于 PCIe 传输速度，所以与 TPU 之间来回传输数据需要 $2 \cdot (BD + DF + BF) / 1.6e10$ 秒。既然我们希望计算耗时比权重加载更长（假设所有权重加载都能与计算重叠），就需要 $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.6e10$。利用 $B \ll D$ 和 $F = 4D$ 这两个假设，可以化简为

$$\frac{8BD^2}{9.2 \times 10^{14}} > \frac{8D^2}{1.6 \times 10^{10}}$$

即

$$B > \frac{9.2 \times 10^{14}}{1.6 \times 10^{10}} \simeq 57{,}500$$

{% enddetails %}

**问题 4 [一般矩阵乘法的延迟]：** 假设我们想把一个权重矩阵 int8[16384, 4096] 与一个大小为 int8[B, 4096] 的激活矩阵相乘，其中 B 是某个未知的批大小。先假设我们在 1 块 TPU v5e 上。

1. 作为 B 的函数，这次乘法需要多长时间？*提示：分别计算从 HBM 加载数组需要多长时间、乘法本身需要多长时间，可能会有帮助。哪一个是你的瓶颈？*
2. 如果我们想从 VMEM 中执行这个操作呢？作为 B 的函数，它需要多长时间？

{% details 点击此处查看答案。 %}

**答案：** (1) 我们需要执行的运算次数为 $2 \cdot 4096 \cdot 16384 \cdot B = 1.3 \times 10^{8} \cdot B$。所以 $T_{\text{计算}} = (1.3 \times 10^{8} \cdot B) / 3.94 \times 10^{14}$ 秒。我们需要从 HBM 向 VMEM 加载 $16384 \cdot 4096 + 4096 \cdot B$ 字节，并从 VMEM 向 HBM 写回 $16384 \cdot B$ 字节。这意味着 $T_{\text{通信}} = (6.7 \times 10^{7} + 2 \times 10^{4} \cdot B) / 8.2 \times 10^{11}$ 秒。假设通信与计算尽可能地重叠，整个乘法大约需要

$$\max\{T_{\text{计算}}, T_{\text{通信}}\} = \max\left\{\frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}, \frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}}\right\}$$

当 $\frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}} < \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}$，或者等价地 $B > 267$ 时，我们是 FLOPs 受限的。这比我们在[第 1 部分](../roofline)中推导出的 240 略大，因为这里我们把 $D$ 和 $F$ 的全部影响都考虑进去了。

(2) 如果改为从 VMEM 加载，我们把 VMEM 到 MXU 的带宽视为 HBM $\leftrightarrow$ VMEM 带宽的 22 倍。这样数据加载项的分母就从 8.2e11 变为 1.80e13，得到 $B > 11$。注意，实际中我们无法把全部 VMEM 带宽都用于加载权重矩阵，所以实际值会更接近 20。

{% enddetails %}

**问题 5 [ICI 带宽]：** 假设我们有一个 TPU v5e `4x4` 切片。假设我们想把一个类型为 `bf16[8, 128, 8192]` 的数组从 `TPU{0,0}` 发送到 `TPU{3, 3}`。假设 TPU v5e 的每跳延迟为 $1\mu s$。

1. 第一个字节多快能到达目的地？
2. 整个传输需要多长时间？

{% details 点击此处查看答案。 %}

**答案：** TPU v5e 是二维连接的。因为我们只有一个 `4x4` 切片（没有大小为 16 的轴），所以没有环绕连接（wraparound）。因此目标芯片有两个端口可以接收数据，源芯片同样有两个端口可以发送数据。我们需要传输的数据量为 `2 * 8 * 128 * 8192 = 1.7e7` 字节。我们可以同时从两个端口传输（即一半数组向右发、一半向下发），因此每秒可传输 `2 * 4.5e10 = 9e10` 字节，这意味着传完整个数组大约需要 `1.7e7 / 9e10 = 188us`（假设是带宽受限的）。在 `4x4` 切片中，芯片 $(0, 0)$ 和 $(3, 3)$ 之间有六跳，因为少于 16 块芯片的轴没有环绕链路。由于每跳的延迟约为 $1\mu s$，第一个字节大约在 `6us` 后到达，而整个传输大约需要 `188 + 6 = 194us`，因为最后一个字节离开源芯片后同样要经过这六跳（一般来说，延迟项和带宽项是相加的，只不过这里延迟只是一个很小的修正）。

{% enddetails %}

**问题 6 [综合运用，较难]：** 设想你有一个大矩阵 **A**：`int8[128 * 1024, 128 * 1024]`，它均匀地分片在一个 TPU v5e 4x4 切片上，但被卸载（offload）到了每块芯片对应的主机 DRAM 中。假设你想把整个数组复制到 TPU{0, 0} 上，并与一个向量 `bf16[8, 128 * 1024]` 相乘。这需要多长时间？*提示：使用上面的数字。*

{% details 点击此处查看答案。 %}

**答案：** 我们先梳理一下需要执行的操作。我们的数组大约 16GB。根据上面的表格，一个 TPU v5e 主机的拓扑是 4x2，所以 4x4 切片有 2 个主机。因此，由于数组是均匀分片的，每个主机实际上持有数组的 1/2，即 8GB。我们需要把这些块全部复制到 TPU{0,0} 上，这有两种选择：

1. 我们可以通过 DCN 复制，然后通过 PCIe 把整个未分片的数组加载到 HBM 中。
2. 我们可以把分片后的数组加载到各自对应的 TPU 上，然后通过 ICI 执行一次 gather，再在 TPU{0,0} 上执行矩阵乘法。

显然选项 (2) 更好。DCN 比 ICI 慢，而且我们更希望通过许多条 PCIe 链路来加载一个大数组，而不是只用少数几条（主机 0 上的 8 条）。下面是系统一部分的示意图。如上所述，注意 TPU 之间通过 ICI 与相邻芯片相连（即使跨主机也是如此），所有 TPU 都（通过 PCIe）连接到各自的主机 CPU，而主机之间通过 DCN 相连。

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="实际上每块芯片都有自己通往主机的 PCIe 链路，不过为了清晰起见，这里只画出了一条。" %}

下面我们来算一算每个环节各需要多长时间：

1. **PCIe 加载**：我们通过 16 条 PCIe 链路加载总共 16GB 的数据块，每条链路的带宽为 `1.6e10` 字节/秒。因此这大约需要 63ms。

2. **ICI 复制：** 现在每块 TPU 持有数组的 16GB / 16 = 1GB。我们的 ICI 带宽为每条链路 9e10 字节/秒（*双向*），从上图可以看出，在这个拓扑中，TPU{0,0} 这块 TPU v5e 的 4 条 ICI 链路中只有 2 条被用到。由于 TPU{0,0} 需要沿 2 个轴、以每条链路 `4.5e10` 字节/秒的速度总共接收 15GB，我们可以得到时间的下界 `15e9 / (4.5e10 * 2) = 167ms`。实际中这可能达不到，因为负载非常不均衡，但大概在 2 倍以内。你会在第 3 部分中看到，执行一次完整的 AllGather 也大约需要 `16e9 / (4.5e10 * 2)`，所以这已经接近最优了。

3. **HBM $\rightarrow$ MXU 加载：** 为了执行最后的矩阵乘法，我们需要通过 HBM 带宽把这 16e9 字节外加 bf16[8, 128 \* 1024] 数组（另外 2MB，可以忽略不计）加载到 MXU 中，这需要 `16e9 / 8.2e11 = 20ms`。

4. **FLOPs：** 我们总共执行 $$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7 \times 10^{11}$$ 次 FLOPs，由于我们每秒能执行 `1.97e14` 次 bf16 FLOPs，所以需要 1.4ms。

总时间的一个上界是上述所有时间之和，但由于 TPU 通常可以让这些操作相互重叠，我们可以把它看作一个由最慢环节决定瓶颈的流水线问题。假设确实如此，那么答案至少是 167ms，考虑到重叠并不完美，很可能接近 200ms。

{% enddetails %}

<h3 markdown=1 class="next-section">第 2 部分到此结束！第 3 部分将介绍划分（partitioning）和跨 TPU 通信，[点击此处](../sharding)继续阅读。</h3>

## 附录 {#appendix}

### 附录 A：TPU 内部的更多细节 {#appendix-a-more-on-tpu-internals}

这里我们将更深入地探讨 TPU 的内部运作。除非另有说明，我们给出的都是 TPU v5p 的规格。

### VPU {#vpu}

VPU 是 TPU 的向量运算核心。VPU 由两部分组成：一个二维 SIMD 向量机（即 **VPU**），负责执行诸如 vadd（向量加法）或 vmax（逐元素取最大值）之类的逐元素算术运算；以及一组称为 **VREGs** 的向量寄存器，用来为 VPU 和 MXU 存放数据。

**VREGs：** 每个 TPU v5p 核心有 64 个 32 位 VREG（TPU v4 中为 32 个），因此每个核心总共约有 `64 * 8 * 128 * 4 = 256kB` 的 VREG 存储（整块芯片则是它的 2 倍，因为有两个核心）。TPU v5p 每个周期可以从 VMEM 加载 3 个寄存器，并向 VMEM 写入 1 个寄存器。

**VPU：** VPU 是一个形状为 `(8, 128)` 的二维向量运算单元，其中大小为 128 的维度称为 lane 轴，大小为 8 的维度称为 sublane 轴。在 v5 上，每个 (lane, sublane) 对包含 4 个相互独立的标准浮点 ALU。VPU 在其每个 ALU 中以一个周期执行大多数算术指令（如 vadd，即向量加法），延迟为 2 个周期，因此例如在 v5 中，你每个周期可以把来自 VREG 的 4 对 f32 值相加。一条典型的 VPU 指令可能形如 `{v2 = vadd.8x128.f32 v0, v1}`，其中 v0 和 v1 是输入 VREG，v2 是输出 VREG。

所有 lane 和 sublane 每个周期都以纯 SIMD 的方式执行同一个程序，但每个 ALU 可以执行不同的运算。因此，例如我们可以在一个周期内处理 1 条 vadd 和 1 条 vsub，每条都作用于两个完整的 VREG，并把输出写入第三个 VREG。

**小测验 [计算 VPU 吞吐量]：** 利用上面的信息，计算 TPU v5p 每秒能执行多少向量 FLOPs。TPU v5p 的时钟频率约为 1.75GHz。

{% details 点击此处查看答案。 %}

*答案*：每个周期，每个核心可以在 `8 * 128` 个 ALU 上执行 4 条向量指令。这样每个核心每周期为 `8 * 128 * 4` FLOPs，即 `8 * 128 * 4 * 1.75e9 = 7e12 FLOPs/s`。注意这比每个核心约 `2e14` 的 MXU FLOPs/s 小得多（大约小 30 倍）。

{% enddetails %}

**归约：** 一般来说，沿 sublane 维度的通信或归约比沿 lane 维度更容易。例如，VPU 支持一种 lane 内的 shuffle 操作，可以在大约一个周期内沿大小为 8 的轴做循环移位（roll）。这可用于沿 sublane 维度高效地做归约（只需分别按 4、2、1 做 shuffle，并做 3 次成对的逐元素求和）。

跨 lane 的归约要困难得多，需要用到一个单独的硬件单元，称为 XLU 或“跨 lane 单元”（cross lane unit），它速度慢且代价相当高。

**与 GPU 的比较：** 对于熟悉 NVIDIA GPU 的读者来说，VPU 中的每个 ALU 类似于一个 CUDA 核心，而单个 VPU lane 类似于一个“Warp Scheduler”，即通常由 32 个 CUDA 核心组成、执行 SIMD 运算的那一组。lane 内的归约相当容易，但如果需要跨 lane，就至少要经过 VMEM/XLU/SMEM，这要慢得多。更多细节请参见 [GPU 部分](../gpus)。

### 标量核心 {#scalar-core}

标量核心是 TPU 的控制单元。它负责取指和分发所有指令，执行从 HBM 到 VMEM 的传输，并且可以通过编程来完成标量元数据方面的工作。由于标量核心是单线程的，其副作用之一是 TPU 的每个核心每个周期只能发起一个 DMA 请求。

为了让你有个概念：单个标量核心控制着一个 VPU（由 4096 个 ALU 组成）、4 个 MXU、2 个 XLU 以及多个 DMA 引擎。单位计算所对应的控制资源如此之少，这种高度倾斜的设计是硬件效率的来源之一，但也限制了以任何有意义的方式进行数据相关向量化的能力。

### 附录 B：脉动阵列是如何工作的？ {#appendix-b-how-does-a-systolic-array-work}

TPU MXU 的核心是一个 `128x128` 的脉动阵列（systolic array；TPU v6e 上为 `256x256`）。在完全饱和时，脉动阵列每 8 个时钟周期可以执行一次 `bf16[8,128] @ bf16[128,128] -> f32[8,128]`<d-footnote>如果你不熟悉这种记法，它的意思是：把一个元素为 bfloat16 的 `8x128` 矩阵与一个元素为 bfloat16 的 `128x128` 矩阵相乘，并把结果存入一个元素为 float32 的 `8x128` 矩阵。</d-footnote>乘法。

* 从本质上说，脉动阵列是一个 `128x128`（`=16,384`）的二维 ALU 网格，每个 ALU 都能执行一次乘加运算。
* 权重（**W**，即 `128x128` 的输入）从上方向下传入（称为 RHS），而输入（**X**，即 `8x128` 的输入）从左侧传入（称为 LHS）。

下面是一个把一组权重（蓝色）与一组激活值（绿色）相乘的简化动画。你会注意到，权重（RHS）先沿对角线方向被部分加载，然后激活值也沿对角线方向被送入。在下面的每一帧中，我们把所有重叠的绿色和蓝色单元相乘，将结果与从上方传入的残余值相加，然后再把结果向下传递一个单元。

{% include figure.liquid path="assets/img/systolic-array.gif" %}

下面是这个动画的一个更一般的版本，展示了输出如何从计算中流出：

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

下图展示了如何在多个 RHS 和 LHS 数组之间进行流水线处理：

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

在加载权重（RHS）和激活值（LHS）时，会有一个初始的流水线气泡。在这个初始气泡之后，新的输入和权重就可以加载进来，而不会产生额外的气泡。

下面是一个做得不太好的 bf16[2, 3] x bf16[3, 3] 矩阵乘法动画，你可以把它想象成一个 2x3 权重矩阵与一个批大小为 1、维度为 3 的输入激活值之间的矩阵乘法。与前面的幻灯片相比，它是旋转过的，输入向右流出而不是向下，但你大致能看出其结构。

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

我们可以高效地对其做流水线处理，从而在不产生过大流水线气泡的情况下完成大矩阵的乘法。话虽如此，重要的是我们的矩阵形状要大于 MXU 的边长，通常是 128x128。一些 TPU（自 TPU v3 起）有多个 MXU，TPU v3 有 2 个，TPU v4/5 有 4 个，所以我们需要确保分块（tiling）的维度大于 128 * MXU 的数量。[这里](https://www.youtube.com/watch?v=sJltBQ4MOHA)有一个很好的相关动画。

Trillium（TPU v6e）有一个 `256x256` 的脉动阵列，这意味着它每个周期可以执行 4 倍的 FLOPs。这也意味着你的张量维度需要是原来的两倍大，才能充分利用 MXU。

[这篇博客文章](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu)中有另一个非常棒的动画，展示了固定权重矩阵下的脉动阵列乘法。
