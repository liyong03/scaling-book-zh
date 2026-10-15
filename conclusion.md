---
layout: distill
title: "结论与延伸阅读"
# permalink: /main/
description: "感谢阅读！这里我们再列出一些可供进一步学习的参考资料。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "../jax-stuff"
previous_section_name: "第 10 部分：JAX"

next_section_url: "../gpus"
next_section_name: "第 12 部分：GPU"

giscus_comments: false
giscus_discussion: 13

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
  - name: "致谢"
    id: acknowledgments
  - name: "延伸阅读"
    id: further-reading
  - name: "反馈"
    id: feedback

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
  .algorithm {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border-style: dashed;
    background-color: #fffaf2;
  }

  .algorithm li {
    margin-bottom: 0px;
  }
---

**感谢你读完全书，恭喜你一路坚持到了最后。** 在结束之前，先说几句致谢：

## 致谢 {#acknowledgments}

这份文档凝聚了 Google DeepMind 许多人的大量心血，我们想在这里简单地向他们致谢！

- James Bradbury、Reiner Pope、Noam Shazeer 和 Blake Hechtman 最早推导出了本书中的许多思想，也是最早从系统视角理解 Transformer 的一批人。
- Sholto Douglas 撰写了本文档的第一版，并发起了这个项目。对于本文档的整体叙事，他的贡献比任何人都大。
- Jacob Austin 主导了将第一版从粗略笔记打磨成一份更精致、更全面的作品的工作。编辑、排版和发布本文档的大部分工作都由他完成，他还协调了其他作者的贡献。
- 大部分图表和动画由 Anselm Levskaya 和 Charlie Chen 制作。
- Charlie Chen 撰写了推理部分，并绘制了许多推理相关的图。
- Roy Frostig 在出版、编辑以及整个过程中的许多其他环节提供了帮助。

我们还要感谢在整个过程中提供了关键反馈的许多其他人，特别是 Zak Stone、Nikhil Sethi、Caitlin Stanton、Alek Dimitriev、Sridhar Lakshmanamurthy、Albert Magyar、Diwakar Gupta、Jeff Dean、Corry Wang、Matt Johnson、Peter Hawkins 以及其他很多人。感谢 Ruiqi Gao 在 HTML 排版上提供的帮助。

**谢谢大家！**

<p markdown=1 class="announce">离开之前，你或许还会喜欢阅读关于 NVIDIA GPU 的全新[第 12 部分](../gpus)！</p>

## 延伸阅读 {#further-reading}

有很多相关的文章，包括以下这些：

- [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html)：一篇很棒的深入剖析 TPU 架构的文章，风格与本书一脉相承。
- [**Domain specific architectures for AI inference**](https://fleetwood.dev/posts/domain-specific-architectures)：一篇对硬件和模型的深入探讨，风格与本书一脉相承。
- [**A Domain-Specific Supercomputer for Training Deep Neural Networks**](https://dl.acm.org/doi/pdf/10.1145/3360307)：最早的 TPU 论文之一，包含许多关于 Google TPU 项目的精彩细节，这些内容本书没有涉及。
- [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html)：一篇更侧重 GPU 和 PyTorch 的教程，讲解 LLM 的 Roofline 分析与性能工程。
- [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)：TPU 编程越来越多地涉及用 Pallas 编写自定义 kernel。这一系列文档讨论了如何编写 kernel，以及许多本书未提及的更底层的 TPU 细节。
- [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM)：虽然专门针对 GPU 和 CUDA，但这是一篇出色的博客文章，展示了如何在 CUDA 中优化矩阵乘法（matmul）kernel。它或许能帮你深入理解 TPU 与 GPU 的区别。
- [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)：这是一份非常好的 JAX 并行 API 指南，也是学习如何实际实现我们在这里讨论的一些思想的好途径。
- [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024)：我们的前同事 Rafi 开过一门很棒的 TPU 性能工程课程，所有幻灯片都放在 GitHub 上。其中很多内容比本书讲得更深入。
- [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102)：一篇详细讨论 Transformer 推理数学的论文，本文档的许多内容都受其启发。
- [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：可以算是本书的 GPU 版对应读物，它更深入地讨论了 PyTorch 在训练中如何实现各种并行技术和节省内存的技术。
- [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/)：一篇博客，包含许多与本书相同的思想，还有一些出色的插图。
- [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework)：一门非常棒的 Stanford 课程，涵盖了 LLM 训练和部署的许多细节，并配有一些有用的练习。其中作业 1 和作业 2 尤其相关。
- [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering)：一份非常实用的 ML 基础设施指南，涵盖了本书没有涉及的主题，比如如何与云服务商谈判、集群管理，以及 GPU 吞吐量的实测数据。
- [**ezyang's blog**](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/)：一位 PyTorch 负责人的博客，内容涵盖分片与 PyTorch 的方方面面，包括一篇 [PyTorch 内部机制指南](https://blog.ezyang.com/2019/05/pytorch-internals/)和一篇[关于分片矩阵乘法的文章](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/)。这里还有很多其他好内容。
- [**The Anatomy of Collective Communication**](https://www.aleksagordic.com/blog/collective-operations)：一篇不错的 GPU 和 TPU 集合通信详解，风格与本书一脉相承。它对 N 维集合通信和 GPU 集合通信的讲解比本书更好。

这一领域仍有大量空间留给全面系统的写作，我们希望本书能激励更多这样的作品！我们也相信这是一个值得学习和研究的、大有可为的领域。在很多情况下，即使手头没有很多硬件加速器，也可以开展这方面的研究。

## 反馈 {#feedback}

欢迎留下评论或问题，帮助我们进一步改进本书。你可以通过
jacobaustin123 [at] gmail [dot] com 联系我们的通讯作者 Jacob Austin，也可以
[在 GitHub 上](https://github.com/jax-ml/scaling-book)提交 issue、pull request 或发起讨论来建议修改。
