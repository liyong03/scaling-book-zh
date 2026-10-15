---
layout: distill
title: "如何理解 GPU"
description: "我们在 Google 很喜欢 TPU，但 GPU 也很棒。本章将深入探讨 GPU 的世界——每块芯片如何工作、芯片之间如何联网，以及这对 LLM 意味着什么，尤其是与 TPU 相比。虽然 NVIDIA、AMD、Intel 等厂商推出了众多 GPU 架构，但这里我们将聚焦于 NVIDIA GPU。本节以<a href='https://jax-ml.github.io/scaling-book/tpus/'>第 2 章</a>和<a href='https://jax-ml.github.io/scaling-book/training'>第 5 章</a>为基础，建议你先阅读这两章。"
date: 2025-08-18
future: true
htmlwidgets: true
hidden: false

section_number: 12

previous_section_url: "../conclusion"
previous_section_name: "第 11 部分：结论"

next_section_url:
next_section_name: "完"

bibliography: main.bib

giscus_comments: false
giscus_discussion: 66

authors:
  - name: Jacob Austin<sup>†</sup>
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: <sup>†</sup>Google DeepMind
  - name: Swapnil Patil<sup>†</sup>
    url: "https://www.linkedin.com/in/swapnil-patil-5b47a068"
  - name:  Adam Paszke<sup>†</sup>
    url: https://x.com/apaszke
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope
    affiliations:
      name: <sup>*</sup>MatX

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "什么是 GPU？"
    id: what-is-a-gpu
  - subsections:
    - name: "内存"
      id: memory
    - name: "GPU 规格汇总"
      id: summary-of-gpu-specs
    - name: "芯片层面的 GPU 与 TPU 对比"
      id: gpus-vs-tpus-at-the-chip-level
    - name: "小测验 1：GPU 硬件"
      id: quiz-1-gpu-hardware
  - name: "网络"
    id: networking
  - subsections:
    - name: "节点层面"
      id: at-the-node-level
    - name: "小测验 2：GPU 节点"
      id: quiz-2-gpu-nodes
    - name: "节点之外"
      id: beyond-the-node-level
    - name: "小测验 3：节点之外"
      id: quiz-3-beyond-the-node-level
  - name: "GPU 上的集合通信是如何工作的？"
    id: how-do-collectives-work-on-gpus
  - subsections:
    - name: "节点内集合通信"
      id: intra-node-collectives
    - name: "跨节点集合通信"
      id: cross-node-collectives
    - name: "小测验 4：集合通信"
      id: quiz-4-collectives
  - name: "GPU 上 LLM 扩展的 Roofline 分析"
    id: rooflines-for-llm-scaling-on-gpus
  - subsections:
    - name: "数据并行"
      id: data-parallelism
    - name: "张量并行"
      id: tensor-parallelism
    - name: "专家并行"
      id: expert-parallelism
    - name: "流水线并行"
      id: pipeline-parallelism
    - name: "示例"
      id: examples
    - name: "GPU 上 LLM 扩展的要点总结"
      id: tldr-of-llm-scaling-on-gpus
    - name: "小测验 5：LLM 的 Roofline"
      id: quiz-5-llm-rooflines
  - name: "致谢与延伸阅读"
    id: acknowledgements-and-further-reading
  - name: "附录"
    id: appendix-a-how-does-this-change-with-gb200
  - subsections:
    - name: "附录 A：GB200 带来了哪些变化？"
      id: appendix-a-how-does-this-change-with-gb200
    - name: "附录 B：更多网络细节"
      id: appendix-b-more-networking-details

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

## 什么是 GPU？ {#what-is-a-gpu}

现代 ML GPU（例如 H100、B200）基本上就是一组专门做矩阵乘法的计算核心（称为 **流式多处理器（Streaming Multiprocessor）**，即 **SM**），连接到一条高速内存（称为 **HBM**）上。下面是示意图：

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" link="true" caption="<b>图：</b>H100 或 B200 GPU 抽象布局示意图。H100 有 132 个 SM，而 B200 有 148 个。我们较宽泛地使用 “Warp Scheduler” 一词，用它来指代一组 32 个 CUDA SIMD 核心<i>以及</i>向它们分派工作的调度器。注意这看起来和 TPU 多么相像！" %}

每个 SM 就像 TPU 的 Tensor Core 一样，拥有一个专用的矩阵乘法核心（不巧的是它也叫 **Tensor Core**<d-footnote>GPU 的 Tensor Core 是 SM 中负责矩阵乘法的子单元，而 TPU 的 TensorCore 则是包含 MXU、VPU 及其他组件的总体单元。</d-footnote>）、一个向量算术单元（称为 **Warp Scheduler**<d-footnote>NVIDIA 并没有给它起一个好名字，所以我们只是在几个糟糕的选项中挑了个最好的来用。Warp Scheduler 主要是向一组 CUDA 核心分派工作的单元，但我们在这里用它来指代控制单元以及它所控制的那组核心。</d-footnote>），以及一块快速的片上缓存（称为 **SMEM**）。TPU 最多只有 2 个独立的 “Tensor Core”，而现代 GPU 则有超过 100 个 SM（H100 上有 132 个）。每个 SM 都远不如 TPU 的 Tensor Core 强大，但系统整体更加灵活。每个 SM 或多或少是完全独立的，所以 GPU 可以同时执行数百个不同的任务。<d-footnote>尽管各个 SM 是独立的，但为了达到峰值性能，它们往往不得不相互协调，因为它们共享一个容量有限的 L2 缓存。</d-footnote>

我们来更详细地看一下 H100 的 SM：

{% include figure.liquid path="assets/gpu/blackwell-sm.png" class="img-small" link="true" caption="<b>图：</b>H100 SM 示意图（<a href='https://wccftech.com/nvidia-hopper-gh100-gpu-official-5nm-process-worlds-fastest-hpc-chip-80-billion-transistors-hbm3-memory/'>来源</a>），展示了 4 个<i>子分区（subpartition）</i>，每个子分区包含一个 Tensor Core、Warp Scheduler、寄存器文件，以及若干组不同精度的 CUDA 核心。靠近底部的 “L1 Data Cache” 就是 256kB 的 SMEM 单元。B200 看起来与之类似，但增加了相当多的 Tensor Memory（TMEM），用来为体量庞大的 Tensor Core 供数。" %}

每个 SM 被划分为 4 个相同的象限，NVIDIA 称之为 **SM 子分区（SM subpartition）**，每个子分区包含一个 Tensor Core、16k 个 32 位寄存器，以及一个称为 Warp Scheduler 的 SIMD/SIMT 向量算术单元，NVIDIA 把它的各个通道（ALU）称为 **CUDA 核心（CUDA Core）**。每个分区的核心组件可以说是 Tensor Core，它执行矩阵乘法，贡献了绝大部分 FLOPs/s，但它并不是唯一值得关注的组件。

* **CUDA 核心：** 每个子分区包含一组称为 CUDA 核心的 ALU，负责 SIMD/SIMT 向量算术。每个 ALU 通常每个周期可以执行 1 次算术运算，例如 f32.add。<d-footnote>较新的 GPU 支持 FMA（融合乘加，Fused-Multiply Add）指令，严格来说每个周期执行两次 FLOPs，NVIDIA 毫不留情地利用这一点把他们公布的规格翻倍。</d-footnote> 每个子分区包含 32 个 fp32 核心（以及数量较少的 int32 和 fp64 核心），它们在每个周期都执行相同的指令。与 TPU 的 VPU 一样，CUDA 核心负责 ReLU、逐点向量运算以及归约（求和）。<d-footnote>历史上，在 Tensor Core 出现之前，CUDA 核心是 GPU 的主要组件，用于渲染，包括光线-三角形求交和着色。在如今的游戏 GPU 上，它们仍然承担着大部分渲染工作，而 TensorCore 则用于上采样（DLSS），这让 GPU 可以以较低分辨率渲染（像素更少 = 工作量更少），再用 ML 进行上采样。</d-footnote>

* **Tensor Core（TC）：** 每个子分区都有自己的 Tensor Core，它是一个专用的矩阵乘法单元，类似于 TPU 的 MXU。Tensor Core 贡献了 GPU 绝大部分的 FLOPs/s（例如在 H100 上，TC 提供 990 bf16 TFLOP/s，而 CUDA 核心只有 66 TFLOPs/s）。
  * [990 bf16 TFLOPs/s](https://www.nvidia.com/en-us/data-center/h100/)，132 个 SM 运行在 1.76GHz，意味着每个 H100 TC 每个周期可以执行 `7.5e12 / 1.76e9 / 4 ~ 1024` 次 bf16 FLOPs，大致相当于一个 8x8x8 的矩阵乘法。<d-footnote>NVIDIA 很少公开 TC 的硬件细节，因此这更多是猜测而非确定的事实——当然，它也说明不了 TC 是如何实现的。我们知道 V100 每个 TC 每个周期可以执行 256 次 FLOPs。A100 可以执行 512 次，H100 可以执行 1024 次，而 B200 的细节虽然没有公布，但很可能约为每 TC 每周期 2048 次 FLOPs，因为 `2250e12 / (148 * 4 * 1.86e9)` 大约是 2048。<a href='https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727'>这里</a>证实了更多细节。</d-footnote>
  * 与 TPU 一样，GPU 可以以更高的吞吐量执行较低精度的矩阵乘法（例如 H100 的 fp8 FLOPs/s 是 fp16 的 2 倍）。低精度训练或推理部署可以显著更快。
  * 自 Volta 以来，每一代 GPU 的 TC 尺寸都比上一代更大（[一篇关于这方面的好文章](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)）。到了 B200，TC 已经大到 SMEM 再也放不下它的输入，因此 B200 引入了一个新的内存空间，称为 TMEM。<d-footnote>在 Ampere 中，Tensor Core 可以由单个 warp 供数，而在 Hopper 中需要一整个 SM（warpgroup），在 Blackwell 中则由 2 个 SM 供数。Blackwell 中的矩阵乘法也变得非常大，以至于其参数（具体来说是累加器）再也放不进寄存器内存/SMEM，所以 Blackwell 增加了 TMEM 来解决这个问题。</d-footnote>

**CUDA 核心比 TPU 的 VPU 更灵活：** GPU 的 CUDA 核心（自 V100 起）使用一种称为 SIMT（*单指令多线程，Single Instruction Multiple Threads*）的编程模型，而 TPU 使用的是 SIMD（*单指令多数据，Single Instruction Multiple Data*）模型。与 TPU VPU 中的 ALU 一样，同一子分区内的 CUDA 核心在每个周期必须执行相同的操作（例如，如果一个核心在做两个浮点数相加，那么该子分区中其他所有 CUDA 核心也必须做同样的事）。但与 VPU 不同的是，每个 CUDA 核心（即 CUDA 编程模型中的 “线程”）都有自己的指令指针，可以被独立地*编程*。当同一个 warp 中的两个线程被指示执行不同的操作时，实际上*两种*操作都会执行，并把不需要执行分歧操作的核心屏蔽掉。

{% include figure.liquid path="assets/gpu/warp-divergence.png" class="img-fluid" caption="<b>图：</b>一组线程中 warp 分歧（warp divergence）的示例（<a href='https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf'>来源</a>）。白色空隙表示至少有一部分物理 CUDA 核心处于停顿状态" %}

这使得线程级别的编程非常灵活，但代价是如果 warp 分歧过于频繁，性能会悄无声息地下降。线程在可访问的内存方面也更灵活：VPU 只能对连续的内存块进行操作，而 CUDA 核心可以访问共享寄存器中的单个浮点数，并维护每个线程各自的状态。

**CUDA 核心的调度也更灵活：** SM 的运行方式有点像多线程 CPU，也就是说它们可以并发地 “调度” 许多程序（**warp**）（每个 SM 最多 64 个），但每个 *Warp Scheduler* 在每个时钟周期只会执行一个程序。<d-footnote>调度到某个 SM 上的 warp 称为 “驻留”（resident）warp。</d-footnote> Warp Scheduler 会自动在活跃的 warp 之间切换，以隐藏内存加载之类的 I/O 操作。相比之下，TPU 通常是单线程的。

### 内存 {#memory}

除了计算单元之外，GPU 还有一个内存层次结构，其中最大的是 HBM（GPU 主内存），然后是一系列更小的缓存（L2、L1/SMEM、TMEM、寄存器内存）。

* **寄存器：** 在 H100/B200 上，每个子分区都有自己的寄存器文件，包含 16,384 个 32 位字（每个 SM `4 * 16384 * 4 = 256kiB`），可供 CUDA 核心访问。
  * 每个 CUDA 核心一次最多只能访问 256 个寄存器，所以尽管每个 SM 最多可以调度 64 个 “驻留 warp”，但如果每个线程使用 256 个寄存器，一次只能容纳 8 个（`256 * 1024 / (4 * 32 * 256)`）。

* **SMEM（L1 缓存）：** 每个 SM 都有自己的 256kB 片上缓存，称为 SMEM，它既可以作为 “共享内存（shared memory）” 由程序员控制，也可以由硬件用作片上缓存。SMEM 用于存储激活值和 TC 矩阵乘法的输入。

* **L2 缓存：** 所有 SM 共享<d-footnote>严格来说，L2 缓存被一分为二，因此在 H100 上，每一半的 SM 各自可以访问 25MB。两半之间有一条链路相连，但带宽较低。</d-footnote>一个相对较大的约 50MB 的 L2 缓存，用于减少对主内存的访问。
  * 它的大小与 TPU 的 VMEM 相近，但要慢**得多**，而且不受程序员控制。这就导致了某种 “超距作用（spooky action at a distance）”：程序员需要调整内存访问模式，才能确保 L2 缓存被充分利用。<d-footnote>L2 缓存由所有 SM 共享这一事实，实际上迫使程序员无论如何都要以相当协调的方式运行各个 SM，尽管原则上它们是独立的单元。</d-footnote>
  * NVIDIA 并未公布其芯片的 L2 带宽，但[实测](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)约为 5.5TB/s。这大约是 HBM 带宽的 1.6 倍，但它是全双工的，所以有效双向带宽接近 3 倍。相比之下，TPU 的 VMEM 要大 2 倍，*而且*带宽高得多（约 40TB/s）。

* **HBM：** GPU 主内存，用于存储模型权重、梯度、激活值等。
  * HBM 容量已从 Volta 的 32GB 大幅增长到 Blackwell（B200）的 192GB。
  * 从 HBM 到 CUDA Tensor Core 的带宽称为 HBM 带宽或内存带宽，在 H100 上约为 3.35TB/s，在 B200 上约为 9TB/s。

### GPU 规格汇总 {#summary-of-gpu-specs}

下面汇总了近期型号的 GPU 规格。同一款 GPU 的不同变体在 SM 数量、时钟频率和 FLOPs 上会有些差异。以下是内存容量数据：

|  GPU  | 架构代际 |   时钟频率   | 每芯片 SM 数 | 每 SM 的 SMEM 容量 | 每芯片 L2 容量 | 每芯片 HBM 容量 |
| :---: | :--------: | :-------------: | :------: | :--------------: | :--------------: | :---------------: |
| V100  |   Volta    | 1.25GHz/1.38GHz |    80    |       96kB       |       6MB        |       32GB        |
| A100  |   Ampere   | 1.10GHz/1.41GHz |   108    |      192kB       |       40MB       |       80GB        |
| H100  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       80GB        |
| H200  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       141GB       |
| B200  | Blackwell  |        ?        |   148    |      256kB       |      126MB       |       192GB       |

所有代际的 GPU 每个 SM 都有 256kB 的寄存器内存。Blackwell 还为每个 SM 额外增加了 256kB 的 TMEM。以下是各芯片的 FLOPs 和带宽数据：

|  GPU  | 架构代际 | 每芯片 HBM 带宽 | 每芯片 FLOPs/s（bf16/fp16） | 每芯片 FLOPs/s（fp8/int8） | 每芯片 FLOPs/s（fp4） |
| :---: | :--------: | :---------: | :----------------------: | :---------------------: | :----------------: |
| V100  |   Volta    |   9.0e11    |            —             |            —            |         —          |
| A100  |   Ampere   |   2.0e12    |          3.1e14          |         6.2e14          |         —          |
| H100  |   Hopper   |   3.4e12    |          9.9e14          |         2.0e15          |         —          |
| H200  |   Hopper   |   4.8e12    |          9.9e14          |         2.0e15          |         —          |
| B200  | Blackwell  |   8.0e12    |          2.3e15          |         4.5e15          |       9.0e15       |

我们没有列出 B100，因为它并未大规模量产。<d-footnote>虽然 NVIDIA 推出过 B100 这一代，但它们只短暂地生产和销售过，据称是因为设计缺陷导致其无法接近所宣称的规格运行。由于发热和功耗问题，它们很难在不降频的情况下达到峰值 FLOPs。</d-footnote> 有些规格会因 GPU 的具体版本而略有不同，因为 NVIDIA GPU 不像 TPU 那样标准化。

下面是一张对比 GPU 和 TPU 组件的实用速查表：

|              GPU              |     TPU     |              它是什么？              |
| :---------------------------: | :---------: | :-----------------------------------: |
| 流式多处理器（Streaming Multiprocessor，SM） | Tensor Core | 包含其他各类单元的核心 “基本单元（cell）” |
|        Warp Scheduler         |     VPU     |      SIMD 向量算术单元      |
|           CUDA 核心           |   VPU ALU   |               SIMD ALU                |
|        SMEM（L1 缓存）        |    VMEM     |       快速片上缓存内存       |
|          Tensor Core          |     MXU     |      矩阵乘法单元       |
|        HBM（又称 GMEM）         |     HBM     |  高带宽、大容量内存  |

### 芯片层面的 GPU 与 TPU 对比 {#gpus-vs-tpus-at-the-chip-level}

GPU 最初是用来渲染电子游戏的，但自 2010 年代深度学习兴起以来，它们越来越像专用的矩阵乘法机器——换句话说，越来越像 TPU。<d-footnote>在深度学习热潮之前，GPU（“图形处理器”，Graphics Processing Unit）做的就是，嗯，图形——主要用于电子游戏。电子游戏用数以百万计的小三角形来表示物体，游戏把这些三角形渲染（或 “光栅化”）成一幅 2D 图像，每秒在屏幕上显示 30-60 次（这个频率称为帧率）。光栅化包括把这些三角形投影到相机坐标系中，并计算哪些三角形覆盖了哪些像素，每秒要进行数十亿次。可以想象，这非常昂贵，而这还只是开始。接下来还要为每个像素着色，把可能与该光线相交的若干个半透明三角形的颜色组合起来。GPU 被设计成能极快地执行这些操作，同时兼顾通用性：你需要同时运行许多不同的 GPU 工作负载（称为 “着色器”，shader），而且没有哪一种操作占主导地位。因此，面向图形的消费级 GPU 可以做矩阵乘法，但那并不是它们的主要功能。</d-footnote> 在某种程度上，这段历史解释了现代 GPU 为什么是现在这个样子。它们并不是纯粹为 LLM 或 ML 模型设计的，而是作为通用加速器设计的，其硬件追求一定程度的 “通用性”，这既是福也是祸。与 TPU 相比，GPU 在应用于新任务时往往更能 “开箱即用”，对优秀编译器的依赖也少得多。但这也使得它们更难分析，也更难达到 Roofline 性能，因为太多编译器特性都可能造成瓶颈。

**GPU 更加模块化。** TPU 有 1-2 个大的 Tensor Core，而 GPU 有数百个小的 SM。同样，每个 TC 有一个大的 VPU，由 4 个可独立编程的 8x128 单元组成（总共 4096 个 ALU）；相比之下，H100 有 132 * 4 = 528 个独立的 SIMD 单元，每个宽 32（总共 16k 个 ALU）。下面是 GPU 与 TPU 的 1:1 对比，突出了这一点：

|              GPU              |           TPU            | H100 数量 | TPU v5p 数量 |
| :---------------------------: | :----------------------: | :----: | :-------: |
| SM（流式多处理器） |       Tensor Core        |  132   |     2     |
|        Warp Scheduler         |        VPU 槽位（slot）         |  528   |     8     |
|        SMEM（L1 缓存）        |           VMEM           |  32MB  |   128MB   |
|           寄存器           | 向量寄存器（VRegs） |  32MB  |   256kB   |
|          Tensor Core          |           MXU            |  528   |     8     |

这种模块化程度上的差异，一方面使 TPU 的制造成本低得多、也更容易理解，但另一方面也给编译器带来了更大的负担，要求它做出正确的处理。由于 TPU 只有单一的控制线程，并且只支持 VPU 宽度的向量化指令，编译器需要手动对所有内存加载和 MXU/VPU 工作进行流水线编排，以避免停顿。GPU 程序员则可以直接启动几十个不同的 kernel，每个都运行在完全独立的 SM 上。但另一方面，这些 kernel 的性能可能会非常糟糕，因为它们造成了 L2 缓存颠簸（thrashing），或者没能合并（coalesce）内存加载；由于运行时的大部分行为都由硬件控制，就很难推断幕后到底发生了什么。因此，TPU 往往能以更少的工作量更接近峰值 Roofline 性能。

**从历史上看，单块 GPU 比同类 TPU 更强大（也更昂贵）：** 单块 H200 的 FLOPs/s 接近 TPU v5p 的 2 倍，HBM 是其 1.5 倍。与此同时，Google Cloud 上 H200 的标价约为 \\$10/小时，而 TPU v5p 约为 \\$4/小时。与 GPU 相比，TPU 通常更依赖于把多块芯片联网在一起。

**TPU 拥有多得多的快速缓存内存。** TPU 的 VMEM 也比 GPU 的 SMEM（+TMEM）多得多，这块内存可用于存储权重和激活值，使它们能够被极快地加载和使用。如果你能持续地把模型权重存储或预取到 VMEM 中，这可以让 TPU 在 LLM 推理上更快。

### 小测验 1：GPU 硬件 {#quiz-1-gpu-hardware}

下面是一些练习题，用来检验上文的部分内容。题目都附有答案，但最好还是先拿起纸笔自己试着解答，再去看答案。

**问题 1 [CUDA 核心]：** 一块 H100 有多少个 fp32 CUDA 核心（ALU）？B200 呢？这与 TPU v5p 中独立 ALU 的数量相比如何？

{% details 点击此处查看答案。 %}

**答案：** H100 有 132 个 SM，每个 SM 有 4 个子分区，每个子分区包含 32 个 fp32 CUDA 核心，因此共有 `132 * 4 * 32 = 16896` 个 CUDA 核心。B200 有 `148` 个 SM，因此总共有 `18944` 个。TPU v5p 有 2 个 TensorCore（通常通过 Megacore 连接），每个 TensorCore 有一个 VPU，具有 (8, 128) 个 lane，每个 lane 有 4 个独立 ALU，因此共有 `2 * 4 * 8 * 128 = 8192` 个 ALU。这大约是 H100 向量 lane 数量的一半，而运行频率大致相同。

{% enddetails %}

**问题 2 [向量 FLOPs 计算]**：一块 H100 有 132 个 SM，时钟频率为 1.59GHz（boost 时最高 1.98GHz）。假设每个 ALU 每个周期可以执行一次向量运算。每秒可以执行多少次向量 fp32 FLOPs？开启 boost 呢？这与矩阵乘法 FLOPs 相比如何？

{% details 点击此处查看答案。 %}

**答案：** `132 * 4 * 32 * 1.59e9 = 26.9TFLOPs/s`。开启 boost 时为 33.5 TFLOPs/s。这只是 [规格表](https://www.nvidia.com/en-us/data-center/h100/) 中所报数值的一半，因为严格来说我们可以在一个周期内完成一次 FMA（融合乘加），它算作两次 FLOPs，但这在大多数情况下并没有什么用。我们可以达到 990 bfloat16 矩阵乘法 TFLOPs/s，因此在忽略 FMA 的情况下，Tensor Core 的 FLOPs/s 大约是 CUDA 核心的 30 倍。

{% enddetails %}

**问题 3 [GPU 矩阵乘法算术强度]：** H100 上 fp16 矩阵乘法的峰值算术强度是多少？B200 呢？fp8 又如何？*这里的算术强度指矩阵乘法 FLOPs/s 与内存带宽之比。*

{% details 点击此处查看答案。 %}

**答案：** 对于 H100，峰值为 990e12 fp16 FLOPs，带宽为 3.35e12 bytes / s。因此临界算术强度为 `990e12 / 3.35e12 = 295`，与 TPU 的 240 相当接近。对于 B200，它是 `2250e12 / 8e12 = 281`，非常接近。这意味着与 TPU 类似，我们需要大约 280 的批大小才能让矩阵乘法达到计算受限。

对于 H100 和 B200，fp8 FLOPs 都恰好是 fp16 的 2 倍，因此峰值算术强度也翻倍，分别为 590 和 562。不过从某种意义上说，如果考虑到权重很可能也会以 fp8 加载，它其实保持不变。

{% enddetails %}

**问题 4 [矩阵乘法运行时间]：** 利用问题 3 的答案，你预计在单块 B200 上执行 `fp16[64, 4096] * fp16[4096, 8192]` 矩阵乘法需要多长时间？`fp16[512, 4096] * fp16[4096, 8192]` 呢？

{% details 点击此处查看答案。 %}

由上可知，当批大小低于 281 个 token 时，我们会处于通信受限状态。因此第一个矩阵乘法完全是带宽受限的。我们需要读写 $2BD + 2DF + 2BF$ 字节（`2*64*4096 + 2*4096*8192 + 2*64*8192=69e6`），带宽为 `8e12` bytes/s，因此大约需要 `69e6 / 8e12 = 8.6us`。实际上我们很可能只能用到总带宽的一部分，所以可能更接近 10-12us。当我们增大批大小时，就完全是计算受限的了，因此预计 `T=2*512*4096*8192/2.3e15=15us`。同样，我们预计只能用到总 FLOPs 的一部分，所以实际可能更接近 20us。

{% enddetails %}

**问题 5 [L1 缓存容量]：** H100 的 L1/SMEM 总容量是多少？寄存器内存呢？这与 TPU 的 VMEM 容量相比如何？

{% details 点击此处查看答案。 %}

**答案：** 每个 SM 有 256kB SMEM 和 256kB 寄存器内存，因此两者各约 33MB（`132 * 256kB`）。合计约 66MB。这大约是现代 TPU 120MB VMEM 的一半，不过 TPU 的寄存器内存总共只有 256kB！TPU VMEM 的延迟低于 SMEM 的延迟，这也是寄存器内存在 TPU 上不那么关键的原因之一（向 VMEM 溢出（spill）和回填（fill）的代价很低）。

{% enddetails %}

**问题 6 [计算 B200 时钟频率]：** NVIDIA 在 [这里](https://resources.nvidia.com/en-us-blackwell-architecture) 报告称 B200 可以执行 80TFLOPs/s 的向量 fp32 计算。已知每个 CUDA 核心在一次 FMA（融合乘加）运算中每周期可以执行 2 次 FLOPs，请估算峰值时钟频率。

{% details 点击此处查看答案。 %}

**答案：** 我们知道共有 148 * 4 * 32 = 18944 个 CUDA 核心，因此每周期可以执行 `18944 * 2 = 37888 FLOPs / cycle`。所以 `80e12 / 37888 = 2.1GHz`，这是一个较高但合理的峰值时钟频率。B200 通常采用液冷，因此更高的时钟频率是比较合理的。

{% enddetails %}

**问题 7 [估算 H100 加法运行时间]：** 利用上面的数据，计算在单块 H100 上把两个 `fp32[N]` 向量相加应该需要多长时间。请同时计算 $T_\text{math}$ 和 $T_\text{comms}$。这个操作的算术强度是多少？如果有条件，也可以试着在 PyTorch 或 JAX 中针对 `N = 1024` 和 `N=1024 * 1024 * 1024` 运行这个操作。结果相比如何？

{% details 点击此处查看答案。 %}

**答案：** 首先，两个 `fp32[N]` 向量相加需要执行 N 次 FLOPs，需要加载 `4 * N * 2` 字节并写回 4 * N 字节，总计 `3 * 4 * N = 12N`。计算二者之比，得到 `total FLOPs / total bytes = N / 12N = 1 / 12`，这相当糟糕。

如上所述，在忽略 FMA 的情况下，开启 boost 时大约可以达到 33.5 TFLOPs/s。但这只有在所有 CUDA 核心都被用上时才成立。对于 `N = 1024`，我们*最多*只能用到 1024 个 CUDA 核心，即 8 个 SM，因此耗时会更长（假设是计算受限，大约会慢 16 倍）。我们的内存带宽为 3.35e12 bytes/s。因此硬件的峰值算术强度为 `33.5e12 / 3.35e12 = 10`。<d-footnote>值得注意的是，这个算术强度在最近几代 GPU 中保持不变。H100 是 33.5 / 3.5，B200 是 80 / 8。原因尚不清楚，但这是一个有趣的现象。</d-footnote> 所以我们会处于严重的通信受限状态。因此运行时间就是

$$T = \max(T_\text{comms}, T_\text{math}) = \frac{12 \cdot N}{\text{3.35e12}} = \frac{N}{\text{2.8e11}}$$

对于 `N = 65,536`，大约是 0.23us。实际上在 JAX 中我们测得的运行时间约为 1.5us，这没有问题，因为我们预计这里会严重受限于延迟。对于 `N = 1024 * 1024 * 1024`，Roofline 估计约为 3.84ms，而实测为 4.1ms，非常好！

{% enddetails %}

## 网络 {#networking}

网络是 GPU 与 TPU 差异最大的领域之一。正如我们所见，TPU 以 2D 或 3D 环面（torus）的方式连接，每个 TPU 只与其相邻的 TPU 相连。这意味着在两个 TPU 之间发送消息必须经过中间的每一个 TPU，并迫使我们在网格上只能使用均匀的通信模式。虽然这在某些方面不太方便，但这也意味着每个 TPU 的链路数量是恒定的，我们可以扩展到任意大的 TPU “Pod”而不损失带宽。

而 GPU 则使用更传统的、基于树的分层交换网络。由 8 个 GPU 组成的集合称为**节点**（GB200 最多可达 72 个<d-footnote>“节点”这个术语有歧义，可以指两种东西：NVLink 域，即通过 NVLink 互连完全连接的一组 GPU；或者连接到同一个 CPU 主机的一组 GPU。在 B200 之前，二者通常是一回事，但在 GB200 NVL72 中，一个 NVLink 域有 72 个 GPU，而每个主机仍然只连接 8 个 GPU。这里我们用“节点”一词指代 NVLink 域，但这一用法存在争议。</d-footnote>），节点内的 GPU 之间通过称为 NVLink 的高带宽互连在 1 跳之内彼此相连；这些节点又通过挂在每个 GPU 上的网卡（NIC），借助带宽较低的 InfiniBand（IB）或以太网连接成更大的单元（称为 **SU**，即可扩展单元，Scalable Unit）。这些单元又可以通过更高层级的交换机连接成任意大的单元。

{% include figure.liquid path="assets/gpu/superpod-diagram.png" class="img-fluid" caption="<b>图：</b>典型 H100 网络的示意图。8 个 GPU 通过 NVSwitch（也称为 NVLink 交换机）连接成一个节点或 NVLink 域，这些节点之间再通过交换式 InfiniBand 网络相连。在 NVLink 域内，每块 H100 约有 450GB/s 的出口带宽，而每个节点进入 IB 网络的出口带宽为 400GB/s。" %}

### 节点层面 {#at-the-node-level}

GPU 节点是一个较小的单元，通常包含 8 个 GPU（GB200 最多可达 72 个），通过全互连（all-to-all）、满带宽、低延迟的 NVLink 互连相连。<d-footnote>有人向我这样描述 NVLink：它有点像加强版的 PCIe 连接，延迟和协议开销都很低，但并不是为可扩展性/容错而设计的；而 InfiniBand 更像以太网，是为更大规模的有损网络设计的。</d-footnote> 每个节点包含若干个高带宽 NVSwitch，负责在所有本地 GPU 之间交换数据包。节点层面的实际拓扑随时间变化很大，包括每个节点的交换机数量；但对于 H100，每个节点有 4 个 NVSwitch，GPU 以 `5 + 4 + 4 + 5` 的链路模式连接到它们，如下图所示：

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>图：</b>从 Pascal（P100）开始的节点（即 NVLink 域）示意图。自 Volta（V100）起，节点内部就通过一组交换机实现了全互连（all-to-all）。H100 节点有 4 个 NVSwitch，通过 25GB/s 的链路连接到全部 8 个 GPU。" %}

对于 Hopper 一代（NVLink 4.0），每条 NVLink 链路具有 25GB/s 的全双工<d-footnote>这里的全双工是指每个方向 25GB/s，两个方向彼此独立。你可以通过该链路总共发送 50GB/s，但每个方向最多 25GB/s。</d-footnote>带宽（B200 为 50GB/s），因此每个 GPU 进入网络的全双工带宽为 `18 * 25=450GB/s`。巨大的 NVSwitch 拥有多达 64 个 NVLink 端口，这意味着带 4 个交换机的 8xH100 节点最多可以处理 `64 * 25e9 * 4=6.4TB/s` 的带宽。下面概述了这些数字随 GPU 代际的变化：

| NVLink 代际 | NVSwitch 代际 | GPU 代际 | NVLink 带宽（GB/s，全双工） | 每 GPU 的 NVLink 端口数 | 节点内 GPU 到 GPU 带宽（GB/s，全双工） | 节点规模（NVLink 域） | 每节点 NVSwitch 数 |
| :--------: | :----------: | :------------: | :----------------------------------: | :----------------: | :------------------------------------------: | :-----------------------: | :-----------------: |
|  **3.0**   |   **2.0**    |     Ampere     |                  25                  |         12         |                     300                      |             8             |          6          |
|  **4.0**   |   **3.0**    |     Hopper     |                  25                  |         18         |                     450                      |             8             |          4          |
|  **5.0**   |   **4.0**    |   Blackwell    |                  50                  |         18         |                     900                      |           8/72            |        2/18         |

Blackwell（B200）的节点包含 8 个 GPU。GB200NVL72 支持更大的、包含 72 个 GPU 的 NVLink 域。我们同时给出了 8 GPU 和 72 GPU 两种系统的细节。

### 小测验 2：GPU 节点 {#quiz-2-gpu-nodes}

下面是更多关于网络的问答题。我发现亲手做这些题特别有用，因为它们会迫使你把实际的通信模式完整地推演一遍。

**问题 1 [H100 节点的总带宽]：** 在带 4 个交换机的 8xH100 节点中，每个节点总共有多少带宽？*提示：*同时考虑 NVLink 和 NVSwitch 的带宽。

{% details 点击此处查看答案。 %}

**答案：** 我们有 4 个第 4 代 NVSwitch，每个具有 `64 * 25e9=1.6TB/s` 的单向带宽。这在交换机层面给出了 `4 * 1.6e12=6.4e12` 的带宽。然而请注意，每个 GPU 只能处理 450GB/s 的单向带宽，这意味着我们最多只有 `450e9 * 8 = 3.6TB/s` 的带宽。由于这个值更小，峰值带宽为 3.6TB/s。

{% enddetails %}

**问题 2 [对分带宽]**：对分带宽（bisection bandwidth）定义为网络的任意均分方式中两部分之间可用带宽的最小值。换句话说，如果把网络分成相等的两半，两半之间能流过多少带宽？你能计算出 8x H100 节点的对分带宽吗？*提示：*对分带宽通常包括两个方向上的流量。

{% details 点击此处查看答案。 %}

**答案：** 任意均分都会使每一半各有 4 个 GPU，每一半都可以向另一半输出 `4 * 450GB/s`。把两个方向的流量都算上，跨越划分的字节量为 `8 * 450GB/s`，即 3.6TB/s 的对分带宽。这与 NVIDIA 报告的数值一致，例如 [这里](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)。

{% enddetails %}

**问题 3 [AllGather 开销]**：给定一个 B 字节的数组，在 8xH100 节点上执行一次（吞吐量受限的）AllGather 需要多长时间？针对 bf16[D<sub>X</sub>, F] 进行计算，其中 `D=4096`、`F=65,536`。*在回答之前，值得先阅读关于 TPU 集合通信的[章节](https://jax-ml.github.io/scaling-book/sharding/)。在这里先思考一下，我们接下来会更详细地讨论集合通信。*

{% details 点击此处查看答案。 %}

**答案：** 每个 GPU 的出口带宽为 450GB/s，每个 GPU 持有 $B / N$ 字节（其中 `N=8`，即节点规模）。我们可以设想每个节点依次把自己的字节发送给其余 $N - 1$ 个节点中的每一个，总共需要 (N - 1) 轮，每轮耗时 $T_\text{comms} = (B / (N * W_\text{unidirectional}))$，即 $T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$。这近似于 $B / W_\text{uni}$，即 $B / \text{450e9}$。

对于给定的数组，`B = 4096 * 65536 * 2 = 536e6` 字节，因此总时间为 `536e6 * (8 - 1) / (8 * 450e9) = 1.04ms`（使用近似式则为 `536e6 / 450e9 = 1.19ms`）。这可能会受限于延迟，因此实际耗时可能更长（实际约为 1.5ms）。

{% enddetails %}

## 节点之外 {#beyond-the-node-level}

在节点层面之外，GPU 网络的拓扑就没那么标准化了。NVIDIA 发布了一个 [DGX SuperPod 参考架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html)，使用 InfiniBand 连接比单个节点更多的 GPU，但客户和数据中心提供商可以根据自身需求自由定制。<d-footnote>例如，Meta 训练 LLaMA-3 所用的数据中心网络与这里的描述有很大不同：它使用以太网、3 层交换网络，并且在顶层使用了超额订阅（oversubscribed）的交换机。</d-footnote>

下面是一个 1024 GPU H100 参考系统的示意图，其中底行的每个方框都是一个 8xH100 节点，包含 8 个 GPU、8 块 400Gbps CX7 网卡（每个 GPU 一块）和 4 个 NVSwitch。

{% include figure.liquid path="assets/gpu/h100-superpod.png" class="img-fluid" caption="<b>图：</b>1024 H100 DGX SuperPod 参考架构示意图，包含 128 个节点（有时是 127 个），每个节点有 8 个 H100 GPU，连接到 InfiniBand 横向扩展（scale-out）网络。每 32 个节点（256 个 GPU）组成的集合称为“可扩展单元”（Scalable Unit，SU）。leaf 和 spine IB 交换机提供了足够的带宽，使节点之间具有满对分带宽。" %}

**可扩展单元：** 每 32 个节点组成的集合称为一个“可扩展单元”（Scalable Unit，SU），挂在同一组 8 个 leaf InfiniBand 交换机之下。这个 SU 有 256 个 GPU，每个节点有 4 个 NVSwitch，另有 8 个 InfiniBand leaf 交换机。图中所示的所有线缆都是 InfiniBand NDR（50GB/s 全双工），配合 64 端口的 NDR IB 交换机（每端口同样是 50GB/s）。*注意，IB 交换机的带宽是 NVSwitch 的 2 倍（64 个端口，每条链路 400 Gbps）。*

**SuperPod：** 整个 SuperPod 再用 16 个顶层“spine” IB 交换机把 4 个这样的 SU 连接起来，从而得到 1024 个 GPU，配有 512 个节点级 NVSwitch、32 个 leaf IB 交换机和 16 个 spine IB 交换机，总计 512 + 32 + 16 = 560 个交换机。leaf 交换机以 32 个节点为一组与节点相连，因此每组 256 个 GPU 对应 8 个 leaf 交换机。所有 leaf 交换机都与所有 spine 交换机相连。

**我们有多少带宽？** InfiniBand 网络（称为“横向扩展网络”，scale-out network）的整体拓扑是一棵**胖树（fat tree）**，其线缆和交换机保证了节点层面以上的满对分带宽（这里为 400GB/s）。这意味着如果把节点分成两半，每个节点都可以同时以 400GB/s 的速率向另一半中的某个节点输出数据。更重要的是，这意味着在横向扩展网络中我们应该能获得大致恒定的 AllReduce 带宽！虽然实际实现未必如此，但你可以设想在横向扩展网络中对任意多个节点执行环形归约（ring reduction），因为总能构造出一个包含所有节点的环。

| 层级 | GPU 数 | 每单元交换机数 | 交换机类型 | 每单元带宽（TB/s，全双工） | GPU 到 GPU 带宽（GB/s，全双工） | 胖树带宽（GB/s，全双工） |
| :---: | :------------: | :-------------------------: | :---------: | :------------------------------------------: | :--------------------------------------: | :---: |
| 节点  |       8        |              4              |     NVL     |                     3.6                      |                   450                    | 450
| Leaf  |      256       |              8              |     IB      |                     12.8                     |                    50                    | 400 |
| Spine |      1024      |             16              |     IB      |                     51.2                     |                    50                    | 400 |

作为对比，TPU v5p 每条链路约有 90GB/s 的出口带宽，沿 3D 环面所有轴合计为 540GB/s 的出口带宽。它不是点对点的，因此只能用于受限的、均匀的通信模式，但它仍然提供了高得多的 TPU 到 TPU 带宽，并且可以扩展到任意大的拓扑（至少可达 8960 个 TPU）。

理论上，GPU 交换网络可以通过增加更多交换机或更多间接层级扩展到任意规模，代价是额外的延迟和昂贵的网络交换机。

<p markdown=1 class="takeaway">**要点**：在 H100 节点内部，每个 GPU 拥有 450GB/s 的满胖树带宽；而在节点之外，节点到节点的带宽降至 400GB/s。这一点对通信原语至关重要。</p>

**GB200 NVL72：** NVIDIA 最近开始生产新的 GB200 NVL72 GPU 集群，它把 72 个 GPU 组合进同一个 NVLink 域，GPU 到 GPU 具有完整的 900GB/s 带宽。这些域又可以连接成更大的 SuperPod，其 IB 胖树带宽也按比例提高（9 倍）。下面是该拓扑的示意图：

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图：</b>一个包含 576 个 GPU 的 GB200 DGX SuperPod 示意图。底层的每个机柜包含 72 个 GB200 GPU。" %}

统计单个节点的出口带宽（上图中的橙色线），我们到 leaf 层有 `4 * 18 * 400 / 8 = 3.6TB/s` 的带宽，是 H100 的 9 倍（正如该节点包含的 GPU 数是 H100 节点的 9 倍）。这意味着关键的节点出口带宽要高得多，*高得多*，而跨节点的集合通信带宽实际上可能比节点内部的*更低*。
更多讨论见 [附录 A](#appendix-a-how-does-this-change-with-gb200)。

|  节点类型  | 每节点 GPU 数 | GPU 出口带宽 | 节点出口带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

<p markdown=1 class="takeaway">**要点**：GB200 NVL72 SuperPod 大幅提高了节点规模以及单个节点的出口带宽，这会显著改变我们的 Roofline 分析。</p>

### 小测验 3：节点之外 {#quiz-3-beyond-the-node-level}

**问题 1 [胖树拓扑]：** 利用上面的 DGX H100 示意图，计算整个 1024 GPU Pod 在节点层面的对分带宽。证明每条链路的带宽都是为保证满对分带宽而选定的。*提示：务必同时计算链路带宽和交换机带宽。*

{% details 点击此处查看答案。 %}

**答案：** 我们逐个组件来分析：

* 首先，每个节点有 8 条 400Gbps NDR IB 线缆连接到 leaf 交换机，使每个节点到 leaf 的带宽为 `8 * 400 / 8 = 400 GB/s`。我们有 8 个 leaf 交换机，每个 3.2TB/s（64 条 400 GBps 链路），但 64 个端口中只有 32 个能用于接收来自 SU 的入口流量，因此 32 个节点共有 `32 * 400 / 8 = 12.8TB/s`，同样恰好是每节点 400GB/s。
* 然后在 spine 层，有 `8 * 16 * 2` 条 400Gbps NDR IB 线缆把每个 SU 连接到 spine，使每个 SU 到 leaf 的带宽为 `8 * 16 * 2 * 400 / 8 = 12.8 TB/s`。同样，这相当于每节点 400GB/s。我们有 16 个 spine 交换机，每个 3.2TB/s，总计 `16 * 3.2 = 51.2 TB/s`，对于 128 个节点而言同样是每节点 400GB/s。

因此，无论以何种方式对分节点，两部分之间每个节点都有 400GB/s 的带宽。每个组件的带宽都恰好满足构成胖树的要求。

{% enddetails %}

**问题 2 [扩展到更大的 DGX Pod]：** 假设我们想在 2048 个 GPU 而不是 1024 个 GPU 上训练。修改上述 DGX 拓扑以支持这一规模，最简单/最好的方法是什么？4096 个呢？*提示：没有唯一正确的答案，但请尽量降低成本。注意链路容量。[这份](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)文档可能会有帮助。*

{% details 点击此处查看答案。 %}

**答案：** 一种选择是保持 SU 结构不变（8 个交换机下挂 32 个节点），只是增加更多 SU，并配备更多顶层交换机。我们需要 2 倍的 spine 交换机，于是会有 8 个 SU 和 32 个 spine 交换机，提供足够的带宽。

这样做的一个问题是每个 leaf 交换机只有 64 个端口，而在上图中我们已经全部用完了。不过，可以很容易地改为每个 spine 只连 1 条 400 Gbps NDR 线缆而不是 2 条，这样总带宽不变，但能节省一些端口。

对于 4096 个 GPU，端口实际上会不够用，因此我们需要再增加一层间接层级，也就是在层次结构中再加一层。NVIDIA 把这些称为“核心交换机”（core switches），并用 128 个 spine 交换机和 64 个核心交换机构建了一个 4096 GPU 的集群。你可以自己算一算，证明这能提供足够的带宽。

{% enddetails %}

## GPU 上的集合通信是如何工作的？ {#how-do-collectives-work-on-gpus}

GPU 可以执行与 TPU 相同的所有集合通信：ReduceScatter、AllGather、AllReduce 和 AllToAll。与 TPU 不同的是，这些操作的工作方式取决于它们是在节点层面（通过 NVLink）还是更高层面（通过 InfiniBand）执行。这些集合通信由 NVIDIA 在 [NVSHMEM](https://developer.nvidia.com/nvshmem) 和 [NCCL](https://developer.nvidia.com/nccl)（读作“nickel”）库中实现。NCCL 已在 [这里](https://github.com/NVIDIA/nccl) 开源。虽然 NCCL 会根据延迟要求/拓扑使用多种不同的实现（[详情](https://github.com/NVIDIA/nccl/issues/1415#issuecomment-2310650081)），但从这里开始，我们将讨论交换式树形网络上的理论最优模型。

### 节点内集合通信 {#intra-node-collectives}

**AllGather 或 ReduceScatter：** 对于节点级的 AllGather 或 ReduceScatter，可以像 TPU 一样沿环执行，每一跳都使用完整的 GPU 到 GPU 带宽。将 GPU 任意排序，然后利用完整的 GPU 到 GPU 带宽把数组的一部分沿环发送。<d-footnote>也可以这样理解：每个 GPU 把自己大小为 $\text{bytes} / N$ 的分块发送给其他 $N - 1$ 个 GPU，总通信量为 $(N - 1) * N * bytes / N$ 字节，得到的结果相同。</d-footnote>每一跳的开销为 $T_\text{hop} = \text{bytes} / (N * \text{GPU egress bandwidth})$，因此总开销为

$$T_\text{AG or RS comms} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU egress bandwidth}} \rightarrow \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

你会注意到这与 TPU 上完全相同。对于 AllReduce，可以像往常一样组合 RS + AG，开销为两倍。

{% include figure.liquid path="assets/gpu/all-gather.gif" class="img-fluid" caption="<b>图：</b>带宽最优的一维环形 AllGather 算法。对于 B 字节，它会经由顶层交换机发送 B / X 字节，共 X - 1 次。" %}

如果你关心延迟（例如数组非常小时），可以做树形归约：先在 2 个一组内做 AllReduce，然后是 4 个、8 个，总共只需 $\log(N)$ 跳而不是 $N - 1$ 跳，不过总开销仍然相同。

<p markdown=1 class="takeaway">**要点：** 在单个节点内对 B 字节的数组做 AllGather 或 ReduceScatter，开销约为 $T_\text{comms} = B * (8 - 1) / (8 * W_\text{GPU egress}) \approx B / W_\text{GPU egress}$。理论上，在 H100 上约为 $B  / \text{450e9}$，在 B200 上约为 $B / \text{900e9}$。除非启用了网内归约，否则 AllReduce 的开销是它的 2 倍。</p>

<b markdown=1 style="color: #57cf57;">小测验 1 [AllGather 耗时]：</b>使用一个具有 450 GB/s 全双工带宽的 8xH100 节点，AllGather(bf16[B<sub>X</sub>, F]) 需要多长时间？设 $B=1024$，$F=16,384$。

{% details 点击此处查看答案。 %}

**答案：** 总共有 $2 \cdot B \cdot F$ 字节，单向带宽为 450e9。这大约需要 $T_\text{comms} = (2 \cdot B \cdot F) / \text{450e9}$，更精确地说是 $(2 \cdot B \cdot F \cdot (8 - 1)) / (8 \cdot \text{450e9})$。代入给定数值，大约为 $(2 \cdot 1024 \cdot 16384) / \text{450e9} = \text{75us}$，更精确地说是 $\text{65us}$。

{% enddetails %}

**AllToAll：** 节点内的 GPU 之间是全互连（all-to-all）的，这让 AllToAll 变得……相当简单。每个 GPU 直接发送到目标节点即可。在节点内，对于 B 字节，每个 GPU 持有 $B / N$ 字节，并向 $N - 1$ 个目标节点各发送 $(B / N^2)$ 字节，总计

$$T_\text{AllToAll comms} = \frac{B \cdot (N - 1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}$$

与之对比，TPU 上的开销为 $B / (4W)$。因此，在单个节点内，我们在运行时间上获得了理论上 2 倍的加速（$B / 4W$ 对比 $B / 8W$）。

对于混合专家（MoE）模型，我们经常需要做*稀疏或不规则（ragged）的 AllToAll*，即保证输出维度上 $N$ 个分片中至多 $k$ 个非零，也就是说 $T_\text{AllToAll} \rightarrow K[B, N]$，其中每个轴上的 $N$ 个条目中至多 $k$ 个非零。其开销会按 $k/N$ 降低，总计约为 $\min(k/N, 1) \cdot B / (W \cdot N)$。对于 MoE，我们通常独立随机地选取非零值，因此有一定概率非零值少于 $k$ 个，于是近似为
$(N-1)/N \cdot \min(k/N, 1) \cdot B / (W \cdot N)$。<d-footnote>真实开销实际上是 $$(1 - \left(\frac{Z - 1}{Z}\right)^K) \cdot \frac{Z - 1}{Z}$$，即 $K$ 次掷骰子中不同结果的期望个数，但它与给出的近似值非常接近。更多细节见附录。</d-footnote>

<b markdown=1 style="color: #c55404ff;">小测验 2 [AllToAll 耗时]：</b>使用一个具有 450 GB/s 单向带宽的 8xH100 节点，AllToAll<sub>X->N</sub>(bf16[B<sub>X</sub>, N]) 需要多长时间？如果已知 8 个条目中只有 4 个非零呢？

{% details 点击此处查看答案。 %}

**答案：** 注意这里 $B$ 是数组的批维度，因此数组总大小为 $V = 2 \cdot B \cdot N$ 字节。由上文可知，在稠密情况下开销为 $V \cdot (N-1) / (W \cdot N^2)$，大约为 $V / (W \cdot N)$。如果已知只有 $\frac{1}{2}$ 的条目是非填充值，我们可以只发送 $V \cdot k/N / (W \cdot N) = V / (2 \cdot W \cdot N)$，大约是总开销的一半。

{% enddetails %}

<p markdown=1 class="takeaway">**要点：** 在单个节点内的 GPU 上对 $B$ 字节的数组做 AllToAll，开销约为 $T_\text{comms} = (B \cdot (8 - 1)) / (8^2 \cdot W_\text{GPU egress}) \approx B / (8 \cdot W_\text{GPU egress})$。对于不规则（top-$k$）AllToAll，开销会进一步降低到 $(B \cdot k) / (64 \cdot W_\text{GPU egress})$。</p>

**实测结果：** 下面是在 8xH100 节点上 AllReduce 带宽的实测结果。Algo BW 是测得的带宽（字节数 / 运行时间），Bus BW 按 $2 \cdot W \cdot (8 - 1) / 8$ 计算，理论上衡量的是实际链路带宽。你会注意到我们确实能达到接近 370GB/s，低于 450GB/s 但相当接近，不过要到每个设备约 10GB 时才行。这意味着，尽管这些估算在理论上是正确的，但需要很大的消息才能真正达到。

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>关闭 SHARP 时 8xH100 节点的 AllReduce 吞吐量。蓝色曲线是实测链路带宽，由实测结果按 $2 * \text{bytes} * (N - 1) / (N * \text{runtime})$ 计算得到。注意，即便是 10GB 的超大数组，我们也没能特别接近宣称的 450GB/s 带宽。" %}

这是个实实在在的问题，因为它让我们能做出的任何理论论断都显著复杂化。例如，即使是对一个大小合理的数组做 AllReduce，比如 LLaMA-3 70B 的 MLP（大小为 `bf16[8192, 28672]`，或在 8 路模型分片下为 `bf16[8192, 3584] = 58MB`），也只能达到约 150GB/s，而峰值为 450GB/s。相比之下，TPU 在小得多的消息大小下就能达到峰值带宽（见附录 B）。

<p markdown=1 class="takeaway">**要点：** 尽管 NVIDIA 宣称 H100 NVLink 的带宽约为 450GB/s，但实践中很难超过 370 GB/s，因此请相应地调整上面的估算。</p>

**网内归约：** 从 Hopper 这一代开始，NVIDIA 交换机支持 [“SHARP”（Scalable Hierarchical Aggregation and Reduction Protocol，可扩展分层聚合与归约协议）](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)，它支持“网内归约”（in-network reductions）。这意味着*网络交换机本身*就能执行归约操作，并将结果复用或“组播”（MultiCast）到多个目标 GPU：

{% include figure.liquid path="assets/gpu/sharp-algorithm.png" class="img-fluid" caption="<b>图：</b>不使用 SHARP 的 AllReduce 理论开销是 2 倍，因为数据必须两次经过每个 GPU。实践中加速只有约 30%（来自 NCCL 2.27.5）。" %}

理论上，这几乎能把 AllReduce 的开销减半，因为每个 GPU 可以把数据发送到顶层交换机，由交换机本身执行归约并把结果广播给每个 GPU，而无需每个 GPU 发出（egress）两次数据，同时还降低了网络延迟。

$$T_\text{SHARP AR comms} = \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

注意这是精确的，并没有相差 $1/N$ 的因子，因为每个 GPU 先发出 $B \cdot (N - 1) / N$，然后接收其本地分片的部分归约结果（流入 $B/N$），完成归约，再次发出 $B/N$，然后接收完全归约后的结果（流入 $B \cdot (N - 1) / N$），最终恰好流入 $B$ 字节。

然而，实践中启用 SHARP 后带宽只提高了约 30%，而预测值是 75%。这只能让有效集合通信带宽达到约 480GB/s，远不到 2 倍。

{% include figure.liquid path="assets/gpu/sharp-all-reduce-cost.png" class="img-fluid" caption="<b>图：</b>节点内启用与不启用 NVIDIA SHARP 时 AllReduce 算法带宽的实测结果。峰值处的收益约为 30% 的吞吐量提升，尽管从算法上讲它本应能达到接近 75% 的提升。" %}

<p markdown=1 class="takeaway">**要点：** 理论上，NVIDIA SHARP（大多数 NVIDIA 交换机都支持）应能将 $B$ 字节 AllReduce 的开销从约 $2 * B / W$ 降到 $B / W$。然而，实践中我们只看到大约 30% 的带宽提升。由于纯 AllReduce 在 LLM 中相当少见，这一特性并不是特别有用。</p>

### 跨节点集合通信 {#cross-node-collectives}

超出节点层级后，开销会更微妙一些。在树上做归约时，可以把它看作自下而上地归约：先在节点内，然后在 leaf 层，再到 spine 层，每一层都使用常规算法。尤其是对于 AllReduce，可以看出这让我们总体上通信更少的数据，因为在节点级做完 AllReduce 之后，我们只需向 leaf 发出 $B$ 字节，而不是 $B * N$ 字节。

**这有多大开销？** 作为一阶近似，由于我们拥有完整的对分带宽（bisection bandwidth），AllGather 或 ReduceScatter 的开销大致等于缓冲区字节数除以节点出口带宽（H100 上为 400GB/s），*与树形归约的任何细节无关*。

$$T_\text{AG or RS comms} = \frac{\text{bytes}}{W_\text{node egress}} \underset{H100}{=} \frac{\text{bytes}}{\text{400e9}}$$

其中 $W_\text{node}$ egress 对于上述 H100 网络通常为 400GB/s（每个节点有 8 条 400Gbps IB 链路出口）。最直观的理解方式是想象在*集群中的每个节点*上做一次环形归约。由于胖树拓扑，我们总能构造一个任意两个节点之间都有 $W_\text{node}$ egress 带宽的环，并做常规归约。节点级归约（几乎）永远不会成为瓶颈，因为它的总带宽更高、延迟更低，不过一般情况下开销为

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network}) = \max\left[\frac{\text{bytes}}{W_\text{GPU egress}}, \frac{\text{bytes}}{W_\text{node egress}}\right]$$

{% details 点击此处查看更精确的推导。 %}

更精确地说，我们实际上是在网络的每一层都做一次环形归约，而这些归约大部分可以重叠，因此有：

$$T_\text{AG or RS comms} = \text{bytes} \cdot max_\text{depth i}\left[\frac{D_i - 1}{D_i \cdot W_\text{link i}}\right]$$

其中 $D_i$ 是深度 $i$ 处的度（即深度 $i$ 处的子节点数），$W_\text{link i}$ 是连接每个子节点到节点 $i$ 的链路带宽。

据此，对于给定拓扑，我们可以把可用的 AllGather/AllReduce 带宽计算为 $min_\text{depth i}(D_i * W_\text{link i} / (D_i - 1))$。在上面的例子中：

* **节点（Node）：** $D_\text{node}$ = 8，因为一个节点中有 8 个 GPU，Wlink i = 450GB/s。因此 AG 带宽为 `450e9 * 8 / (8 - 1) = 514GB/s`。
* **Leaf：** $D_\text{leaf}$ = 32，因为一个 SU 中有 32 个节点，Wlink i = 400GB/s（8 条 400Gbps IB 链路）。因此带宽为 `400e9 * 32 / (32 - 1) = 413GB/s`。
* **Spine：** $D_\text{spine}$ = 4，因为有 4 个 SU，$W_\text{link i}$ = 12.8TB/s（来自上文的 `8 * 16 * 2 * 400Gbps` 条链路）。带宽为 `12.8e12 * 4 / (4 - 1) = 17.1TB/s`。

因此总的 AG 或 RS 带宽为 `min(514GB/s, 413GB/s, 17.1TB/s) = 413GB/s`，瓶颈在 leaf 层，所以实践中 $T_\text{AG or RS comms} = B / \text{413GB/s}$，也就是说，即使在最高层，我们也有约 413GB/s 的 AllReduce 带宽。对于使用 SHARP 的 AllReduce，带宽会比这略低（约 400GB/s），因为没有 $(N - 1) / N$ 因子。不过，450GB/s 和 400GB/s 已经足够接近，可以用作近似值。

{% enddetails %}

**其他集合通信：** 除非启用 SHARP，否则 AllReduce 的开销仍是上面的 2 倍。NVIDIA 也销售支持 SHARP 的 IB 交换机，但并非所有云服务商都配备。AllToAll 在跨节点时变化很大，因为它们不像 AllReduce 那样是“分层的”。如果我们想把数据从每个 GPU 发送到其他每个 GPU，就无法利用节点级的完整对分带宽。这意味着，如果一个 N 路 AllToAll 跨越 $M = N / 8$ 个节点，开销为

$$T_\text{AllToAll comms} = \frac{B \cdot (M - 1)}{M^2 \cdot W_\text{node egress}} \approx \frac{B}{M \cdot W_\text{node egress}}$$

其有效带宽实际只有 50GB/s 而不是 400GB/s。我们从单个 H100 节点内的 $B / (8 * \text{450e9})$ 变成跨越 2 个节点时的 $B / (2 \cdot \text{400e9})$，性能下降超过 4 倍。

下面是 1024-GPU DGX H100 SuperPod 架构的汇总：

|   层级   | GPU 数量 | 度（子节点数） | 交换机带宽（全双工，TB/s） | 线缆带宽（全双工，TB/s） | 集合通信带宽（GB/s） |
| :-------: | :------------: | :-----------------: | :----------------------------------: | :---------------------------------: | :-------------------------: |
|   节点    |       8        |          8          |                 6.4                  |                 3.6                 |             450             |
| Leaf（SU） |      256       |         32          |                 25.6                 |                12.8                 |             400             |
|   Spine   |      1024      |          4          |                 51.2                 |                51.2                 |             400             |

我们用“集合通信带宽”（Collective Bandwidth）这一术语来描述 GPU 或节点能够发出数据的有效带宽。它也等于 $\text{bisection bandwidth} * 2 / N$。

<p markdown=1 class="takeaway">**要点：** 超出节点层级后，对 B 字节做 AllGather 或 ReduceScatter 的开销大致为 $B / W_\text{node egress}$，在 H100 DGX SuperPod 上为 $B / \text{400e9}$；除非启用 SHARP，否则 AllReduce 的开销是其两倍。整体拓扑是胖树，旨在为任意两个节点之间提供恒定的带宽。</p>

**数组沿另一个轴分片时的归约：** 考虑如下归约的开销

$$\text{AllReduce}_X(A[I_Y, J]\ \{ U_X \})$$

其中我们要 AllReduce 的数组本身沿另一个轴 $Y$ 分片。在 TPU 上，与未分片版本相比，该操作的总开销降低为 $1 / Y$，因为每个轴发送的数据量只有原来的 $1 / Y$。在 GPU 上，开销取决于哪个轴是“内层”轴（节点内还是节点间），以及每个分片是否跨越多于一个节点。假设 $Y$ 是内层轴，数组总共有 $\text{bytes}$ 字节，则总开销实际上会降低 $Y$ 倍，但前提是 $Y$ 跨越多个节点：

$$T_\text{comms at node} = \frac{\text{bytes}}{W_\text{GPU egress}} \cdot \frac{1}{\min(Y, D_\text{node})}$$

$$T_\text{comms in scale-out network} = \frac{\text{bytes}}{W_\text{node egress}} \cdot \frac{D_\text{node}}{\max(D_\text{node}, Y)}$$

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network})$$

其中 N 是 GPU 数量，$D_\text{node}$ 同样是一个节点中的 GPU 数量（节点的度）。可以看到，如果 $Y < D_\text{node}$，我们在节点级获得了收益，但通常看不到总运行时间的减少；而如果 $Y > D_\text{node}$，我们获得的加速与跨越的节点数成正比。

如果想精确描述环形归约，对于树形的 AllGather<sub>X</sub>(A<sub>Y</sub> { U<sub>X</sub> })（假设 Y 是内层轴），一般规则为

$$T_\text{AR or RS comms} = \text{bytes} \cdot \max_{\text{depth } i}\left[\frac{D_i - 1}{D_i \cdot \max(Y, S_{i-1}) \cdot W_{\text{link } i}}\right]$$

其中 $S_i$ 是 M * N * …，即树中第 i 层以下子节点的规模。这大致是说，我们跨越的 GPU 或节点越多，可用带宽就越大，但仅限于该节点内部。

**小测验 3 [沿 2 个轴分片]：** 假设我们要在单个 SU（256 个芯片）上执行 $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$，其中 $Y$ 是内层轴。作为 $D$、$F$ 和 $Y$ 的函数，这需要多长时间？

{% details 点击此处查看答案。 %}

**答案：** 我们可以把它分成两种情况：Y <= 8 和 Y > 8。当 $Y <= 8$ 时，我们仍然受限于 leaf 交换机，所以答案照常为 $T_\text{comms} = 2 * D * F * (32 - 1) / (32 * 400e9)$。当 Y > 8 时，由上文可得，大致为

$$T_\text{comms} = \frac{2 \cdot D \cdot F \cdot 256}{Y \cdot \text{12.8e12}} = \frac{2DF}{Y \cdot \text{50GB/s}}$$

对于 `D = 8192`、`F = 32,768`，我们有：

{% include figure.liquid path="assets/gpu/sharded-all-gather-cost.png" class="img-fluid" caption="<b>图：</b>随着内层轴跨越更多节点，分片 AllGather 的理论开销。" %}

注意，如果我们恰好做 8 路模型并行，确实会把节点级归约的开销降低为原来的 1/8，但总开销保持不变；所以它是免费的，却无助于提升整体带宽。

{% enddetails %}

<p markdown=1 class="takeaway">**要点：** 当我们有多个分片轴时，外层归约的开销会按内层轴所跨越的节点数成比例降低。</p>

### 小测验 4：集合通信 {#quiz-4-collectives}

**问题 1 [SU AllGather]：** 只考虑一个包含 M 个节点、每个节点 N 个 GPU 的 SU。在一次 AllGather 中，节点级交换机精确地流入和流出多少字节？顶层交换机呢？

{% details 点击此处查看答案。 %}

**答案：** 我们逐步分析归约的各个组成部分：

1. 每个 GPU 向交换机发送 $B / MN$ 字节，总流入为 $NB / MN = B / M$ 字节。
2. 我们把全部 $B / M$ 字节向上发出到 spine 交换机。
3. 我们从 spine 交换机流入 $B * (M - 1) / M$ 字节
4. 我们发出 $B - B / MN$ 字节 $N$ 次，总计 $N * (B - B / MN) = NB - B / M$。

总计流入 $B$、流出 $BN$，因此瓶颈应在流出上，总时间为 $T_\text{AllGather} = BN / W_\text{node} = B / \text{450e9}$。

对于 spine 交换机，计算实际上更简单。必须流入 $B / M$ 字节 M 次（总计 $B$ 字节），然后流出 $B (M - 1) / M$ 字节 $M$ 次，总流出为 $B * (M - 1)$。由于这一项大得多，开销为 $T_\text{AllGather} = B \cdot (M - 1) / (M \cdot W_\text{node}) = B \cdot (M - 1) / (M \cdot \text{400e9})$。

{% enddetails %}

**问题 2 [单节点 SHARP AR]：** 考虑一个包含 N 个 GPU 的单节点。在使用 SHARP（网内归约）的 AllReduce 中，交换机精确地流入和流出多少字节？

{% details 点击此处查看答案。 %}

**答案：** 和之前一样，我们逐步分析。

1. 每个 GPU 发送 $B * (N - 1) / N$ 字节，因此流入为 $N * B * (N - 1) / N = B * (N - 1)$。
2. 我们累加部分和，并向每个 GPU 发回 $B / N$ 字节，因此流出 $N * B / N = B$ 字节。
3. 我们在本地对剩余部分做部分求和，然后发回交换机。总计流入 $N * B / N = B$ 字节。
4. 我们收集所有分片并组播出去，向 $N$ 个目的地各发送 $B * (N - 1) / N$，总流出为 $B * (N - 1) / N * N = B * (N - 1)$。

因此总计流入和流出均为 $B * (N - 1) + B = BN$ 字节。这印证了整体吞吐对应的耗时恰好为 $B / W_\text{egress}$。

{% enddetails %}

**问题 3 [跨节点 SHARP AR]：** 考虑一个在 N 个 GPU 的单节点上分片的数组 bf16[D<sub>X</sub>, F<sub>Y</sub>]。AllReduce(bf16[D, F<sub>Y</sub>] { U<sub>X</sub> }) 需要多长时间？可以假设我们使用网内归约。如果有多于一个节点，情况有何不同？

{% details 点击此处查看答案。 %}

**答案：** 我们可以尝试修改上一题的答案。基本上，我们先从每个 GPU 发出 $B * (X - 1) / XY$ 字节，然后向每个 GPU 发回 $B / XY$，再把同样数量的数据发回交换机，然后向每个 GPU 发回 $B * (X - 1) / XY$。总流入和流出为 $NB / Y$，因此总时间为 $T_\text{comms} = NB / (Y * N * W_\text{link}) = N * 2DF / (Y * N * W_\text{link}) = 2 * D * F / (Y * W_\text{link})$，所以总时间确实随 $Y$ 减小。

如果超出单个节点，我们可以做与上面大致相同的归约，但从节点级交换机发出时，需要发送全部 B 字节，而不只是 $B / Y$。这是因为我们需要让每个分片保持独立。

{% enddetails %}

**问题 4 [Spine 级 AR 开销]：** 考虑与上面相同的设定，但 $Y = 256$（因此 AR 发生在 spine 层）。AllReduce 需要多长时间？同样可以假设使用网内归约。

{% details 点击此处查看答案。 %}

**答案：** 这让我们可以利用 spine 层那高得离谱的带宽。在 4 个 SU 上，spine 层有 51.2TB/s 的带宽，即每个 SU 12.8TB/s。使用 SHARP，这最少只需 `2 * D * F / 12.8e12` 秒。

{% enddetails %}

**问题 5 [2 路 AllGather 开销]：** 计算在恰好 2 个节点上对 $B$ 字节做 AllGather 的精确开销。*务必计算精确开销而不是近似值，并同时考虑节点内和跨节点的开销。*

{% details 点击此处查看答案。 %}

**答案：** 在节点级，有 $T_\text{comms} = B * 7 / (8 * \text{450e9}) = B / \text{514e9}$，而节点之外实际上是 $T_\text{comms} = B * (2 - 1) / (2 * \text{400e9}) = B / \text{800e9}$。因此，我们实际上受限于节点级归约，而不是 leaf 层！这正是 DeepSeek v3 等采用 2 路数据并行的动机。

{% enddetails %}

## GPU 上 LLM 扩展的 Roofline 分析 {#rooflines-for-llm-scaling-on-gpus}

现在来看前面所有内容的最终目标：理解 GPU 上 LLM 扩展的 Roofline。这是对 TPU 训练章节（[此处](../training)）的补充。和那一章一样，这里的目标是考察不同并行策略下的总 $T_\text{math}$ 和 $T_\text{comms}$，并弄清楚在什么情况下 $T_\text{comms} > T_\text{math}$。和之前一样，我们只考虑 MLP 块，其运算为

$$\text{MLP}(x) \equiv x[B, D] *_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D]$$

其中 $B$ 是**以 token 计**的全局批大小（即 $B = \text{batch size} \cdot \text{sequence length}$）。

这里我们重新列出上面那张展示 GPU 级和节点级有效带宽的表格：

|  节点类型  | 每节点 GPU 数 | GPU 出口带宽 | 节点出口带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

**注意：** GPU 出口带宽和节点出口带宽都会决定 LLM 的 Roofline。我们用 $W_\text{collective}$ 来表示 GPU 带宽或节点带宽，具体取决于我们是在节点内还是节点之上进行操作。

我们像对 TPU 所做的那样，来看**数据并行、张量并行、流水线并行、专家并行**及其组合的计算-通信 Roofline。本节余下部分在具体计算时将聚焦于 H100 的 Roofline。GB200-NVL72 的 Roofline 总体相同，但由于其节点出口带宽更大，有时瓶颈反而会出现在节点级。

### 数据并行 {#data-parallelism}

如前所述，数据并行（DP）和 ZeRO 分片在反向传播中要么涉及一次权重 AllReduce，要么涉及一次 ReduceScatter + AllGather。由于这两者开销相同，对于*不使用网内归约*的纯数据并行或 FSDP，要做到计算受限，在反向传播中每一层、轴大小为 X 时，有：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{W_\text{collective}}$$

因此，要使 $T_\text{math} > T_\text{comms}$，需要 $B / (XC) > 1 / W_\text{collective}$，即 

$$\frac{B}{X} > \frac{C}{W_\text{collective}}$$

其中 $W_\text{collective}$ 是 GPU 级或节点级的出口带宽，取决于我们是在节点内还是跨节点分片。因此：

* **在节点内**，只需每个 GPU 的 **token** 批大小 > $\text{990e12} / \text{450e9} = 2200$。
* **在 SU 内或 spine 层**，BS > $\text{990e12} / \text{400e9} = 2475$。

这比 TPU 上高出不少——TPU 上使用全部三个轴时这个数字是 850。例如，在 16000 张 H100 上训练的 LLaMA-3 需要至少 40M token 的批大小（作为参考，他们实际用的是 16M）。DeepSeek v3 在 2048 张 H800 GPU 上训练，带宽更低，只有 300GB/s（而 H100 为 450GB/s），因此每个 GPU 需要 $\text{990e12} / \text{300e9} = 3300$ 个 token，即总共约 6.7M（实践中他们用的是 4M）。

在启用网内归约并使用纯数据并行时，理论上 AllReduce 带宽翻倍，这会让上面两个数字都减半。然而，实践中的收益更接近 30%，这基本上只是弥补了我们通常难以达到标称数值的差距。此外，由于纯数据并行很少有用，这在实践中基本无关紧要。

**MoE 模型：** 对于有 E 个专家、每个 token 激活 k 个专家的混合专家（MoE）模型，上式变为

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W_\text{collective}}$$

这会把每个 GPU 的 token 批大小放大 $E/k$ 倍，即

$$\frac{B}{X} > \frac{E}{k} \frac{C}{W_\text{collective}}$$

例如，对于新的 OpenAI 开源模型（$k=4$、$E=128$），跨节点时这个数字增加到 `32 * 2475  = 79,200`，高得有点离谱。

**X 很小时会怎样？** 当我们只做例如 2 节点的数据并行时，可以受益于 $(X - 1) / X$ 的缩放，得到

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N * C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (X-1)}{X \cdot W_\text{collective}}$$

其中 X 是节点数，$N = 8 \cdot X$。于是对于稠密模型，有 $B / N > \alpha \cdot (X - 1) / X$，例如 $B / N > \text{1237}$，是上面数值的一半。正因如此，你会相当频繁地看到 2 路数据并行。

<p markdown=1 class="takeaway">**要点：** 假设完美重叠且 FLOPs 利用率完美，数据并行和 ZeRO 分片在 H100 或 B200 上需要每个 GPU 约 2500 个 token 的批大小才能做到计算受限。对于 MoE 模型，这一数字会增大 $E / k$ 倍，即总参数量与激活参数量之比。当数据并行的规模较小时，临界批大小会降低。</p>

### 张量并行 {#tensor-parallelism}

张量并行需要对激活值做一次 AllGather 和一次 ReduceScatter，我们需要让它们与 MLP 的 FLOPs 重叠。换句话说，在前向传播中，我们有

$$T_\text{math} = \frac{2\cdot 2 \cdot BDF}{Y \cdot C}$$

$$T_\text{comms} = \frac{2\cdot 2 \cdot BD}{W_\text{collective}}$$

要保持计算受限，就得到如下规则

$$Y < \frac{F \cdot W_\text{collective}}{C}$$

在节点内，这大约是 $F / 2200$，跨节点则约为 $F / 2475$。对于像 LLaMA-3 那样 $F=\text{28000}$ 的情况，这大约是 11 路 TP（向下取整则约为 8 路，正好是一个节点的大小）。和上面一样，当我们恰好跨越 2 个节点时会额外获得 2 倍带宽，所以一般可以做 16 路张量并行（$F > 2475 \cdot (Y - 8)$），理论上最多能做到 19 路模型并行。

<p markdown=1 class="takeaway">**要点：** 在大小为 Y 的轴上做张量并行、前馈维度为 F 时，一旦 $Y > F / 2475$ 就会变成通信受限，这通常把我们限制在节点内 TP，或最多 2 个节点的 TP。</p>

### 专家并行 {#expert-parallelism}

正如上文已经提到的，混合专家（MoE）模型的权重是原来的 E 倍，而 FLOPs 只增加 k 倍，这让数据并行变得困难得多。我们可以沿专家维度对权重分片来在一定程度上缓解这一问题，即 W<sub>in</sub>[E<sub>Z</sub>, D, F]。为了完成 MLP 块，我们需要引入 2 次 AllToAll，把激活值发送到对应的专家。

如上所述，如果这次 AllToAll<sub>Z->k</sub>([B, D, k]) 跨越多个节点，其开销大约为 $T_\text{AllToAll} = 2 \cdot B \cdot D \cdot (Z-8)/Z \min(8 * k / Z, 1)$，因此对于纯专家并行，我们需要

$$T_\text{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

$$T_\text{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W \cdot Z} \cdot \min\left(\frac{8 \cdot k}{Z}, 1\right)$$

我们要么需要 $K > Z/8$ 且 $F > \alpha \cdot (Z - 8)/k$，要么需要 $Z \gg K$ 且 $F > 8 \cdot \alpha$，其中 $\alpha = C/W$。这给出了专家并行可行的两个区间：一个是少量专家并行（大约 2 个节点）且 $F$ 较小，另一个是 $F$ 较大且 $Z$ 可以任意大（最多 E 路专家并行）。

这两种情况在实践中都能见到：要么是少量专家并行（比如 DeepSeek v3，它的 F 非常小，跨节点专家并行的规模也相对较小且受限），要么是 F 较大的模型，此时我们可以在做 TP 的同时做大规模的跨节点 EP。

<p markdown=1 class="takeaway">**要点：** 如果 $F < 8 * C / W_\text{node}$，专家并行可以跨越 1-2 个节点，开销与 TP 相近（略低）；如果 $F > 8 * C / W_\text{node}$，我们可以以相对较低的开销做大规模的专家并行（最多跨 $E$ 个节点）。</p>

### 流水线并行 {#pipeline-parallelism}

流水线并行把各层拆分到不同节点上，通信开销极低，因为我们只是每隔几层发送一次小规模微批次的激活值。历史上流水线一直受“流水线气泡”之苦，但借助新的零气泡（zero-bubble）流水线方法，通常可以避免气泡。

流水线的总体通信开销非常小：若有 $N_\text{MB}$ 个微批次和 $N_\text{stages}$ 个阶段，则每跳开销为 $T_\text{comms per hop} = 2 \cdot B \cdot D / (W \cdot N_\text{MB})$，共有 $N_\text{MB} + N_\text{stages} - 2$ 跳，因此大致为

$$T_\text{total PP comms} = \frac{2BD}{W \cdot N_\text{MB}} \cdot (N_\text{MB} + N_\text{stages} - 2)$$

$$T_\text{per-layer comms} \approx 1.5 \cdot \frac{2BD}{W \cdot N_\text{layers}}$$

由于要除以 $N_\text{layers}$，这比其他任何开销都小得多。换句话说，从通信角度看，流水线几乎是免费的。那我们为什么不直接用流水线呢？原因有以下几点：

(1) **代码复杂度：** 流水线不像其他方法那样能很好地融入自动并行框架（比如 XLA 的 GSPMD）。由于它引入了微批次来隐藏流水线气泡，会改变程序的结构；而定制的零气泡流水线调度需要对前向和反向传播做复杂的交错编排，进一步加剧了这个问题。

(2) **流水线让数据并行和 FSDP 变得困难：** 不做流水线的最大原因可能是它与 FSDP 和数据并行配合得很差。尤其是 ZeRO-3 分片效果很差，因为它要求我们在每个微批次上都对权重做 AllGather，而我们只有 $B / N_\text{microbatches}$ 个 token 来摊销 AllGather 的开销，这是行不通的。此外，在反向传播期间，*在最后一个微批次通过某个阶段之前，我们无法对梯度做 AllReduce 或 ReduceScatter，这意味着会有大量无法重叠的通信时间。*

{% include figure.liquid path="assets/gpu/pipeline-bubble.png" class="img-fluid" caption="<b>图：</b>一个 2 阶段、2 个微批次的流水线示例。F 表示某一阶段的前向传播，B 表示某一阶段的反向传播（开销为前向的 2 倍）。G 表示数据并行的 AllReduce，它可能比单个微批次的耗时长得多。" %}

(3) **流水线气泡与步骤不均衡：** 从上面这个（糟糕的）流水线调度中可以看出，朴素的流水线调度很容易产生大量气泡（即浪费的计算）。在上图中，第二阶段在第 0 步空闲，第一阶段在第 2 步到第 3 步空闲，第二阶段在最后一步又处于空闲。虽然通过精心调度可以在一定程度上避免这些气泡，但通常仍会有一些。我们还必须在关键路径上把激活值从一个阶段传给下一个阶段，这会增加额外开销：

{% include figure.liquid path="assets/gpu/pipeline-transfer.png" class="img-fluid" caption="<b>图：</b>一个流水线示例，红色表示传输开销。它使各阶段之间相互错开，并增加了流水线气泡的开销。" %}

上述每个问题都有变通办法，但往往实现复杂、难以维护；不过相对于其他方法，流水线仍然是一种通信开销很低的技术。

**关于延迟的提醒：** 如前所述，即使消息相当大，GPU 也很难达到完整的 AllReduce 带宽。这意味着，即使理论上我们可以把例如专家并行的 AllToAll 扩展到多个节点，实际上可能连总带宽的 50% 都难以达到。因此我们确实会尽量把 TP 或 EP 限制在较少的节点内，以尽量减小延迟开销。

### 示例 {#examples}

**DeepSeek 是怎么做的？** 作为参考，[DeepSeek V3](https://arxiv.org/abs/2412.19437) 使用 2048 块 H800 GPU 训练，配置如下：

* 64 路专家并行（EP），跨越 8 个节点
* 16 路流水线并行（PP）
* 2 路 ZeRO-1 数据并行（DP）

他们的稳态批大小为 `4096 * 15360 = 62,914,560` 个 token，即每块 GPU 30k 个 token。可以看到这已经相当大了，但他们的模型也非常稀疏（k=8，E=256），所以需要相当大的批大小。可以看到，在 64 路 EP 和 16 路 PP 下，我们总共得到 1024 路模型并行，这意味着 AllReduce 是在 spine 层完成的；又因为它只有 2 路，实际上我们获得了 $2 / (2 - 1) = 2$ 倍的带宽。这也有助于降低最后那次数据并行 AllReduce 与最后几个流水线阶段重叠时的开销。

**LLaMA-3 是怎么做的？** LLaMA-3 在 16k 块 GPU 上以 16M token 的批大小训练，即每块 GPU 约 1k 个 token。他们的配置是：

* 节点内 8 路张量并行（TP）
* 16 路流水线并行（PP）
* 128 路 ZeRO-1 数据并行

这也是一个稠密模型，所以总体上这些都相当简单。16 路 PP 把数据并行 AllReduce 的开销降低了 16 倍，这有助于降低临界批大小。

### GPU 上 LLM 扩展的要点总结 {#tldr-of-llm-scaling-on-gpus}

让我们退一步，对目前学到的内容做一个总体总结：

* **数据并行或 FSDP（ZeRO-1/3）需要每块 GPU 约 2500 个 token 的本地批大小**，不过理论上网络内归约（in-network reduction）+ 纯 DP 可以在一定程度上降低这个数值。
* **张量并行在约 8 路以内是计算受限的**，但我们没有足够的带宽在变成通信受限之前扩展到远超这个规模。这基本上把我们限制在单个 NVLink 域内（即单节点，或者需要使用最多 72 块 GPU 的 GB200NVL72）。
* **任何跨越多个节点的模型并行形式都能进一步降低 FSDP 的开销**，因此我们常常希望混合使用 PP + EP + TP 来跨越多个节点并降低 FSDP 开销。
* **如果你能应对零气泡流水线的代码复杂度，并保持相当大的批大小以避免数据并行瓶颈，流水线并行效果会很好。** 流水线通常使 ZeRO-3 无法使用（因为你需要在每个流水线阶段都做 AllGather），但你可以改用 ZeRO-1。

**从宏观上看，这给出了在 GPU 上对大模型分片的一套方案：**

* 对于相对较小的稠密模型，如果批大小足够，激进的 FSDP 效果很好，必要时可以辅以一定程度的流水线或张量并行。
* 对于更大的稠密模型，1-2 个节点的 TP + 多节点 PP + 纯 DP 的某种组合效果很好。
* 对于 MoE，上述规则同样适用，但我们还可以做专家并行，一般来说我们更倾向于专家并行而不是 TP。如果 $F > 8 * C / W_\text{node}$，我们可以做大量的多节点专家并行；否则大约只能做 2 个节点的 EP。

### 小测验 5：LLM 的 Roofline {#quiz-5-llm-rooflines}

**问题 1 [B200 Roofline]：** B200 DGX SuperPod（**不是 GB200 NVL72**）的节点内带宽翻了一倍（900GB/s 出口带宽），但横向扩展网络中的带宽相同（400GB/s）（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html)）。总 FLOPs 见上文。这会如何改变模型并行和数据并行的 Roofline？

{% details 点击此处查看答案。 %}

**答案：** 我们在 bfloat16 下的 FLOPs/s 从 990 增加到 2250 TFLOPs，提高了 2.25 倍。由于带宽变为 2 倍，在节点内我们的 Roofline 大致保持不变。例如对于 TP，临界算术强度上升到 `2250e12 / 900e9 = 2500`，所以限制为 $Y < F / 2500$，只是略高一点（而且除非节点规模增大，否则这对我们没有帮助）。

然而在节点之外，由于没有额外的带宽，我们反而更难保持计算受限了！例如对于数据并行，临界批大小增加到 `2250e12 / 400e9 = 5625`，因为我们的 GPU 能在相同带宽下完成多得多的 FLOPs。

拥有 72-GPU 节点的 GB200 SuperPod 通过增加更多出口带宽改变了这一点（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)）。

{% enddetails %}

**问题 2 [如何对 LLaMA-3 70B 分片]：** 考虑 LLaMA-3 70B，使用 bfloat16 训练，Adam 优化器状态为 fp32。

1. 仅仅为了存储权重和优化器状态，我们至少需要多少块 H100？
2. 假设我们想在 4096 块 H100 GPU 上训练 15T 个 token，并且达到了 45% 的 MFU（模型 FLOPs 利用率）。训练需要多长时间？
3. LLaMA-3 70B 的 `F = 28,672`，训练时批大小约为 4M 个 token。在不变成通信受限的前提下，我们最多能做多大规模的模型并行？用它加上纯 DP，能否在 4k 块芯片上训练 LLaMA-3 并保持计算受限？用 ZeRO-3 呢？用 8 路流水线呢？*注意：同时考虑通信开销和 GPU 内存占用。*

{% details 点击此处查看答案。 %}

1. 权重需要 2 字节，优化器状态需要 8 字节，所以至少需要 700GB。每块 GPU 有 80GB DRAM，因此最少需要 9 块 GPU，或者（向上取整）至少 2 个 8xH100 节点。这样训练会花上极长的时间，而且也放不下梯度检查点，但这是一个下界。
2. 这总共需要 `6 * 70e9 * 15e12 = 6.3e24 bf16 FLOPs`。每块 GPU 能做 `990e12` FLOPs，所以在 45% MFU 下我们能达到 1.8e18 FLOPs/s。因此整个训练需要 3.5e6 秒，即 40 天。
3. 在节点内，我们有 450GB/s 的带宽，所以限制大约是 `F / 1995 = 28672 / 1995 = 14.372`。由于这不跨越 2 个节点，现实中意味着我们最多做 8 路模型并行。
   1. 这样就需要做 512 路 DP。首先，我们要看内存是否足够。由于模型只分片了 8 路，这意味着 `700GB / 8 = 87.5GB / GPU`，放不下，所以不行！
   2. 使用 ZeRO-3 和 8 路 TP，我们将做 512 路 ZeRO-3。这不会有任何内存问题，因为我们对所有内容都做了激进的分片。每块 GPU 的批大小为 `4e6 / 4096 = 976`。这相当低，甚至低于纯 DP 的限制，而由于还需要移动权重，ZeRO-3 的限制是纯 DP 的两倍。所以不行。
   3. 使用 8 路流水线后，每个模型并行分片现在跨越 8 个节点。正如我们所见，这把 leaf 层 AllGather 的开销降低到原来的 1/8，因此那里的整体 AllReduce/AllGather 带宽从 400GB/s 变为 `8 * 400GB/s = 3200GB/s`。此时 Roofline 为 `990e12 / 3200e9 = 309`，所以应该没问题！我们只需要高效地实现流水线即可。

{% enddetails %}

**问题 3 [Megatron-LM 超参数]：** 考虑这张来自 [Megatron-LM 仓库](https://github.com/NVIDIA/Megatron-LM) 的图，它展示了他们很高的 MFU 数值。

{% include figure.liquid path="assets/gpu/megatron-hparams.png" class="img-fluid" %}

注意他们的序列长度处处都是 4096。对于 16B、70B 和 314B 模型，每块 GPU 的 token 批大小是多少？假设数据并行是最外层的轴，并假设使用 bfloat16 归约，判断其中每一个在理论上是计算受限还是通信受限，以及是否存在更优的配置？

{% details 点击此处查看答案。 %}

**答案：** 我们先从每块 GPU 的批大小算起。

* **16B**：每块 GPU `192 * 4096 / 192 = 4096` 个 token
* **70B**：每块 GPU `384 * 4096 / 768 = 2048` 个 token
* **314B**：每块 GPU `1536 * 4096 / 3072 = 2048` 个 token

这意味着除了第一个之外，其余都在每批约 2k 个 token 附近徘徊，这恰好接近我们为 FSDP 计算出的临界阈值。我们曾基于 spine 层归约算出这个界限为 2,472 token / GPU，在这里大致应该会起作用。不过对于 70B 和 314B，由于我们分别有 16 路和 64 路模型（PP + TP）分片，在 spine 层分别获得了 2 倍和 8 倍更好的吞吐量，这意味着它们分别在大约 1k 和 300 token / step 时就应该是计算受限的。

{% enddetails %}

## 致谢与延伸阅读 {#acknowledgements-and-further-reading}

本章在很大程度上得益于许多学识渊博的 GPU 专家的帮助，包括：

* Adam Paszke，他帮助解释了在 GPU 上编写 kernel 的实际情况。
* Swapnil Patil，他最早解释了 GPU 网络是如何工作的。
* Stas Bekman，他指出 GPU 的实际经验表现往往与宣称的规格不同。
* Reiner Pope，他帮助厘清了 GPU 和 TPU 在硬件层面的对比。
* Frédéric Bastien，他对芯片层面的内容给出了详细反馈。
* Nouamane Tazi，他在 GPU 上训练 LLM 的经验帮助改进了 Roofline 一节。
* Sanford Miller，他帮助我理解了 GPU 是如何组网的，以及 NVIDIA 的规格与实际部署中常见情况的对比。

关于 GPU 有大量优秀的读物，以下是我最喜欢的一些：

* [SemiAnalysis' History of the NVIDIA Tensor Core](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)：一篇精彩的文章，描述了 GPU 如何从视频游戏引擎转变为 ML 加速器。
* [SemiAnalysis' Analysis of Blackwell Performance](https://semianalysis.com/2024/04/10/nvidia-blackwell-perf-tco-analysis/)：值得一读，可以帮助理解下一代 NVIDIA GPU。
* [H100 DGX SuperPod Reference](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)：内容枯燥但很有用，介绍了较大的 GPU 集群是如何组网的。[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)是一份关于 GB200 系统的类似文档。
* [Hot Chips Talk about the NVLink Switch](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)：关于 NVLink 和 NCCL 集合通信的有趣读物，尤其涵盖了网络内归约。
* [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)：一份大型半开放 LLM 训练报告的好例子，描述了他们如何选择分片配置。
* [How to Optimize a CUDA Matmul](https://siboehm.com/articles/22/CUDA-MMM)：一篇很棒的博客，描述了如何使用 CUDA 核心实现高效的矩阵乘法，并关注 GPU 上的缓存一致性。
* [HuggingFace Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：一份关于 GPU 上 LLM 并行的指南，本章部分受其启发。
* [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)：一份更侧重 GPU 和 PyTorch 的教程，讲解 LLM 的 Roofline 和性能工程。
* [Cornell Understanding GPU Architecture site](https://cvw.cac.cornell.edu/gpu-architecture)：一份与本书类似的指南，更具体地比较了 GPU 和 CPU 的内部结构。

## 附录 A：GB200 带来了哪些变化？ {#appendix-a-how-does-this-change-with-gb200}

Blackwell 引入了一系列重大的网络变化，包括总体 NVLink 带宽翻倍（900GB/s）的 NVLink 5。B200 仍然是 8-GPU 节点，和 H100 一样，但 GB200 系统（将 B200 GPU 与 Grace CPU 结合）引入了大得多的 NVLink 域（NVL72 中为 72 块 GPU，理论上最多可达 576 块）。更大的 NVLink 域也实际上提高了节点出口带宽，从而降低了节点层以上的集合通信开销。

{% include figure.liquid path="assets/gpu/b200-node.png" class="img-small" caption="<b>图：</b>展示 GB200 NVL72 单元构成方式的示意图，包含 18 个交换机和 72 块 GPU。" %}

在节点内，这一带宽提升（从 450GB/s 到 900GB/s）并没有带来太大差别，因为每块 GPU 的总 FLOPs/s 也翻了一倍。我们的 Roofline 大体保持不变，不过由于 NVLink 带宽好得多，专家并行变得更容易了。

在节点之外，变化更大。下面是一张来自[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)的 SuperPod 示意图。

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图：</b>展示一个由 576 块 GPU 组成的 GB200 DGX SuperPod 的示意图。" %}

可以看到，每个节点的出口带宽增加到 `4 * 18 * 400 / 8 = 3.6TB/s`，而 H100 中只有 400GB/s。这使有效的跨节点 Roofline 改善了约 4 倍（因为每块芯片的 FLOPs 也翻了一倍）。现在我们可能要开始担心瓶颈是否出现在节点层，而不是横向扩展层。

**Grace Hopper：** NVIDIA 还销售 GH200 和 GB200 系统，它们把一定数量的 GPU 与一颗 Grace CPU 搭配在一起。例如，一个 GH200 包含 1 块 H200 和 1 颗 Grace CPU，而一个 GB200 系统包含 2 块 B200 和 1 颗 Grace CPU。这种系统的一个优势是 CPU 通过全带宽的 NVLink 连接（称为 NVLink C2C）与 GPU 相连，因此 CPU 到 GPU 的带宽非常高，适合把参数卸载（offload）到主机内存。换句话说，对于任意一块 GPU，访问主机内存的带宽与访问另一块 GPU 的 HBM 完全相同。

## 附录 B：更多网络细节 {#appendix-b-more-networking-details}

下面是一个 NVLink 4 交换机的示意图。它总共有 64 个 NVLink4 端口（每个端口使用 2 条物理通道），以及一个负责通道间交换的大型交叉开关（crossbar）。相比之下，TPU 使用的是带有可动态重新配置反射镜的光交换机。

{% include figure.liquid path="assets/gpu/nvlink4.png" class="img-fluid" caption="<b>图：</b>单个 NVLink4 交换机的更底层视图。" %}

在每一层，我们都可能受限于可用的链路带宽或交换机总带宽。

* **节点层：** 在节点层，我们有 4 * 1.6TB/s = 6.4TB/s 的 NVSwitch 带宽，但 8 块 GPU 中的每一块只能以 450GB/s 的速率向交换机出口传输，这意味着节点内的实际峰值带宽为 450e9 * 8 = 3.6TB/s（全双工）。
* **SU/leaf 层：** 在 SU 层，我们有 8 个交换机以全互连（all-to-all）的方式通过 1x400 Gbps Infiniband 连接 32 个节点。这为我们提供了 8 * 32 * 400 / 8 = 12.8TB/s 的节点出口带宽，而交换机层面有 8 * 1.6TB/s = 12.8TB/s，两者恰好一致。
* **Spine 层：** 在 spine 层，我们有 16 个交换机通过 2x400 Gbps 链路连接 32 个 leaf 交换机，因此有 32 * 16 * 400 * 2 / 8 = 51.2TB/s 的出口带宽。与 leaf 交换机不同，spine 交换机的全部 64 个端口都朝下，因此每个交换机可以传输 64 * 400 / 8 = 3.2TB/s 的流量，在交换机层面得到 16 * 3.2TB/s = 51.2TB/s，同样恰好一致。

平均到每块 GPU，这意味着节点层的 GPU 到 GPU 带宽为 450GB/s，SU 层和 spine 层均为 50GB/s。

**GPU 实测 AR 带宽：**

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>8xH100 集群上的 AllReduce 带宽（节点内，禁用 SHARP）。" %}

TPU v5p 带宽（1 个轴）：

{% include figure.liquid path="assets/gpu/tpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>TPU v5p 4x4x4 集群上的 AllReduce 带宽（沿一个轴）。" %}

下面还有 AllGather 带宽：

{% include figure.liquid path="assets/gpu/gpu-all-gather-bw.png" class="img-fluid" caption="<b>图：</b>8xH100 集群上的 AllGather 带宽（节点内）。" %}

{% include figure.liquid path="assets/gpu/tpu-all-gather-bw.png" class="img-fluid" caption="<b>图：</b>TPU v5e 8x16 集群上的 AllGather 带宽（沿一个轴）。" %}

**关于 AllToAll 开销的更多内容：**

这里我们可以把近似值 $\min(K / Z) * (Z - 1) / Z$ 与真实值 $(1 - ((Z - 1) / Z) ** K) * (Z - 1) / Z$ 进行比较。除了 $Z$ 较小的情况外，两者很接近。

{% include figure.liquid path="assets/gpu/all-to-all-approx.png" class="img-fluid" caption="<b>图：</b>随着分片数增加，非均匀（ragged）AllToAll 的近似开销与真实开销的比较。" %}
