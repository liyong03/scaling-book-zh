---
layout: distill
title: "如何扩展你的模型"
subtitle: "从系统视角看 TPU 上的 LLM"
# permalink: /main/
description: "训练 LLM 常常让人感觉像炼金术，但理解和优化模型的性能却不必如此。本书旨在揭开语言模型扩展这门科学的神秘面纱：TPU（以及 GPU）如何工作、彼此之间如何通信，LLM 如何在真实硬件上运行，以及在训练和推理时如何并行化你的模型，使其在超大规模下高效运行。如果你曾经好奇过“训练这个 LLM 应该花多少钱”、“自己部署这个模型需要多少内存”或者“AllGather 是什么”，我们希望这本书能对你有所帮助。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: false
giscus_discussion: 2

section_number: 0

previous_section_url: ""
previous_section_name: "第 0 部分：引言"

next_section_url: roofline
next_section_name: "第 1 部分：Roofline 模型"

bibliography: main.bib

citation: true

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
  - name: 总体概览
    id: high-level-outline
  - name: 各章节链接
    id: links-to-sections

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
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

{% include figure.liquid path="assets/img/dragon.png" class="img-fluid" %}

深度学习的很多东西至今仍可归结为某种黑魔法，但优化模型的性能却不必如此——即使在超大规模下也是如此！一些相对简单的原则处处适用——从单个加速器到数万个加速器都是如此——而理解这些原则能让你做很多有用的事情：

- 粗略估计模型各部分距离其理论最优还有多远。
- 在不同规模下，对不同的并行方案（即如何把计算拆分到多个设备上）做出有依据的选择。
- 估算训练和运行大型 Transformer 模型所需的成本和时间。
- 设计能够利用[特定](https://arxiv.org/abs/2205.14135)[硬件](https://arxiv.org/abs/1911.02150)[特性](https://arxiv.org/abs/2007.00072)的算法。
- 基于对当前算法性能瓶颈的明确理解来设计硬件。

**预备知识：**我们假设你对 LLM 和 Transformer 架构有基本了解，但不一定了解它们在大规模下如何运行。你应该了解 LLM 训练的基础知识，最好对 JAX 也有一些基本的熟悉。一些有用的背景阅读材料包括关于 Transformer 架构的[这篇博客文章](https://jalammar.github.io/illustrated-transformer/)以及[最初的 Transformer 论文](https://arxiv.org/abs/1706.03762)。另外也可以看看[这份清单](conclusion#further-reading)，里面有更多适合同步阅读和后续阅读的有用材料。

**目标与反馈：**读完本书后，你应该能够比较自如地估算出在给定硬件平台上 Transformer 模型的最佳并行方案，以及训练和推理大致需要多长时间。如果做不到，请给我们发邮件或留言！我们很想知道怎样才能把内容讲得更清楚。

<p markdown=1 class="announce">你可能也会喜欢阅读关于 NVIDIA GPU 的新增[第 12 部分](gpus)！</p>

### 为什么你应该关心这些？ {#why-should-you-care}

三四年前，我认为大多数 ML 研究人员并不需要理解本书中的任何内容。但如今，即使是“小”模型也运行得如此接近硬件极限，以至于做新颖的研究要求你必须考虑大规模下的效率问题。<d-footnote>从历史上看，ML 研究在系统创新和软件改进之间大致遵循着一种 tick-tock 式的交替循环。Alex Krizhevsky 当年不得不编写极其丑陋繁琐的 CUDA 代码来让 CNN 跑得更快，但短短几年之内，Theano 和 TensorFlow 这样的库就让你不必再这样做了。也许这里也会发生同样的事，本书中的所有内容在几年后都会被抽象掉。但扩展定律（scaling laws）一直把我们的模型推向硬件能力的最前沿，而且在可预见的未来，做前沿研究似乎很可能与理解如何高效地将模型扩展到大型硬件拓扑密不可分。</d-footnote> **如果以 Roofline 效率下降 20% 为代价，那么在基准测试上取得 20% 的提升也毫无意义。**有前景的模型架构经常失败，要么是因为它们*无法*在大规模下高效运行，要么是因为没有人投入精力让它们做到这一点。

**“模型扩展”的目标是：在增加用于训练或推理的芯片数量的同时，让吞吐量获得成比例的线性增长。**这被称为“*强扩展*（strong scaling）”。虽然增加更多芯片（即“并行”）通常会减少计算时间，但代价是芯片之间会增加额外的通信。当通信耗时超过计算耗时，我们就变成了“通信受限”，无法实现强扩展。<d-footnote>随着计算时间的减少，你通常还会在单芯片层面遇到瓶颈。你那闪亮的新 TPU 或 GPU 标称每秒可执行 500 万亿次运算，但如果不小心，它同样很容易只发挥出十分之一的性能——因为它忙于在内存中来回搬运参数。单芯片计算能力、内存带宽和总内存之间的相互作用，是扩展问题的关键所在。</d-footnote> 如果我们对硬件足够了解，能够预判这些瓶颈会出现在哪里，就可以设计或重新配置模型来避开它们。<d-footnote>硬件设计者面对的是相反的问题：构建能为我们的算法提供恰好足够的计算、带宽和内存的硬件，同时尽量降低成本。你可以想象这种“协同设计”问题有多让人焦虑：你必须押注于第一批芯片真正可用时（通常是 2 到 3 年之后）算法会是什么样子。TPU 的故事是这场博弈中一次巨大的成功。矩阵乘法是一种独特的算法，因为它每字节内存所使用的 FLOPs 远多于几乎任何其他算法（每字节 N 次 FLOPs），而早期的 TPU 及其脉动阵列（systolic array）架构在其问世时实现了远优于当时 GPU 的性价比（perf / $）。TPU 是为 ML 工作负载而设计的，而带有 Tensor Core 的 GPU 也正在迅速转变以填补这一细分领域。但你可以想象，如果神经网络没有流行起来，或者以某种 TPU（本质上不如 GPU 灵活）无法应对的根本性方式发生了变化，那代价会有多大。</d-footnote>

*本书的目标是解释 TPU（以及 GPU）硬件如何工作，以及 Transformer 架构如何演进以便在当前硬件上表现良好。我们希望本书既能帮助设计新架构的研究人员，也能帮助致力于让当前一代 LLM 跑得更快的工程师。*

## 总体概览 {#high-level-outline}

本书的整体结构如下：

[第 1 部分](roofline)讲解 Roofline 分析，以及哪些因素会限制我们的扩展能力（通信、计算和内存）。[第 2 部分](tpus)和[第 3 部分](sharding)详细讨论 TPU 如何工作，既包括作为单个芯片的情形，也包括——这一点至关重要——作为一个由带宽和延迟有限的芯片间链路连接起来的互连系统的情形。我们将回答以下问题：

* 一个特定大小的矩阵乘法应该花多长时间？在什么情况下它会受限于计算、内存带宽或通信带宽？
* TPU 是如何连接在一起组成训练集群的？系统的每个部分各有多少带宽？
* 在多个 TPU 之间收集（gather）、分散（scatter）或重新分布数组需要多长时间？
* 如何高效地对以不同方式分布在各设备上的矩阵做乘法？

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>图：</b>来自<a href='tpus'>第 2 部分</a>的一张示意图，展示了 TPU 如何执行逐元素乘积。取决于数组的大小和各条链路的带宽，我们可能处于计算受限（充分利用硬件的全部计算能力）或内存受限（瓶颈在于内存加载）的状态。" %}

五年前，ML 领域的架构百花齐放——ConvNet、LSTM、MLP、Transformer——但现在基本上只剩下 Transformer<d-cite key="transformers"></d-cite>了。我们坚信，理解 Transformer 架构的每一个组成部分都是值得的：每个矩阵的确切大小、归一化发生在哪里、每个部分有多少参数和 FLOPs<d-footnote>即浮点运算（FLoating point OPs），基本上就是所需加法和乘法的总次数。虽然许多资料用 FLOPs 表示“每秒运算次数”，但我们用 FLOPs/s 来明确表示这一含义。</d-footnote>。[第 4 部分](transformers)会仔细讲解这些“Transformer 数学”，展示如何计算训练和推理中的参数量和 FLOPs。这能告诉我们模型会使用多少内存、我们会在计算或通信上花多少时间，以及注意力相对于前馈块何时会变得重要。

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图：</b>一个标准的 Transformer 层，其中每个矩阵乘法（matmul）都用圆圈中的一个点表示。所有参数（不包括归一化层）都以紫色显示。<a href='transformers'>第 4 部分</a>会更详细地讲解这张图。" %}

[第 5 部分：训练](training)和[第 7 部分：推理](inference)是本书的核心，我们在其中讨论一个根本性问题：给定一个特定大小的模型和一定数量的芯片，我该如何并行化模型，使其保持在“强扩展”区间内？这是一个简单的问题，答案却出人意料地复杂。总体来说，有四种主要的并行技术用于把模型拆分到多个芯片上（**数据并行**、**张量并行**、**流水线并行**和**专家并行**），还有许多其他技术用于降低内存需求（**重计算**、**优化器/模型分片（又称 ZeRO）**、**主机卸载（host offload）**、**梯度累积**）。我们会在这里讨论其中的许多技术。

我们希望读完这些部分后，你能够针对新的架构或场景自行在这些技术之间做出选择。[第 6 部分](applied-training)和[第 8 部分](applied-inference)是实践教程，将这些概念应用于流行的开源模型 LLaMA 3。

最后，[第 9 部分](profiling)和[第 10 部分](jax-stuff)介绍如何在 JAX 中实现其中一些想法，以及在出现问题时如何对代码做性能分析和调试。[第 12 部分](gpus)是新增的一部分，深入探讨了 GPU。

我们在全书中都尽量提供一些练习题供你自己动手解答。请不要有压力，不必读完所有部分，也不必按顺序阅读。也请给我们留下反馈。目前这还是一份草稿，会持续修订。谢谢！

*我们要感谢 James Bradbury 和 Blake Hechtman，本书中的许多想法都出自他们。*

<h3 markdown=1 class="next-section">闲话少说，[这里是关于 TPU Roofline 的第 1 部分](roofline)。</h3>

## 各章节链接 {#links-to-sections}

*这个系列可能比实际需要的要长，但我们希望这不会让你望而却步。前三章是预备知识，如果你已经熟悉相关内容可以跳过，不过它们引入了后文会用到的记号。最后三部分可能是最具实用价值的，因为它们讲解了如何处理真实的模型。*

**第一篇：预备知识**

* [**第 1 章：Roofline 分析简介**](roofline)。算法受三方面限制：计算、通信和内存。我们可以借助它们来估算算法的运行速度。

* [**第 2 章：如何理解 TPU**](tpus)。TPU 是如何工作的？这又如何影响我们能训练和部署哪些模型？

* [**第 3 章：分片矩阵及其乘法**](sharding)。在这里，我们通过我们最喜欢的运算——（分片的）矩阵乘法——来讲解模型分片和多 TPU 并行。

**第二篇：Transformer**

* [**第 4 章：你需要知道的所有 Transformer 数学**](transformers)。一个 Transformer 在前向传播和反向传播中要用多少 FLOPs？你能算出它的参数量吗？它的 KV 缓存有多大？我们在这里把这些数学一一推导清楚。

* [**第 5 章：如何为训练并行化 Transformer**](training)。FSDP。Megatron 分片。流水线并行。给定一定数量的芯片，我该如何以给定的批大小尽可能高效地训练给定大小的模型？

* [**第 6 章：在 TPU 上训练 LLaMA 3**](applied-training)。我们会如何在 TPU 上训练 LLaMA 3？需要多长时间？要花多少钱？

* [**第 7 章：关于 Transformer 推理的一切**](inference)。模型训练好之后，我们还得部署它。推理引入了一个新的考量因素——延迟——并改变了内存方面的格局。我们将讨论分离式部署（disaggregated serving）如何工作，以及如何思考 KV 缓存。

* [**第 8 章：在 TPU 上部署 LLaMA 3**](applied-inference)。在 TPU v5e 上部署 LLaMA 3 要花多少钱？延迟与吞吐量之间如何权衡？

**第三篇：实践教程**

* [**第 9 章：如何对 TPU 代码做性能分析**](profiling)。真实的 LLM 从来不像上面的理论那样简单。在这里，我们讲解 JAX + XLA 技术栈，以及如何使用 JAX/TensorBoard 性能分析器（profiler）来调试和修复实际问题。

* [**第 10 章：用 JAX 为 TPU 编程**](jax-stuff)。JAX 提供了一系列用于并行化计算的神奇 API，但你需要知道如何使用它们。这里有有趣的示例和练习题。

**第四篇：结论与附加内容**

* [**第 11 章：结论与延伸阅读**](conclusion)。结束语，以及关于 TPU 和 LLM 的延伸阅读。

* [**第 12 章：如何理解 GPU**](gpus)。关于 GPU 的附加章节：它们如何工作、如何组网，以及它们的 Roofline 与 TPU 有何不同。
