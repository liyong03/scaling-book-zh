---
layout: distill
title: "在 TPU 上训练 LLaMA 3"
# permalink: /main/
description: "让我们仔细看看，如何利用上一节学到的内容在 TPU v5p 上训练 LLaMA 3 模型。它们有多大？不同配置下训练的开销有多大？它们是如何分片的？我们来做一些粗略估算，看看前几节的内容如何对应到真实的模型上。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 6

previous_section_url: "../training"
previous_section_name: "第 5 部分：训练"

next_section_url: ../inference
next_section_name: "第 7 部分：推理"

bibliography: main.bib

giscus_comments: false
giscus_discussion: 8

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
  - name: "LLaMA 3 长什么样？"
    id: what-does-llama-3-look-like
  - name: "计算参数量和 FLOPs"
    id: counting-parameters-and-flops
  - name: "如何为训练分片 LLaMA 3-70B"
    id: how-to-shard-llama-3-70b-for-training
  - name: "练习题"
    id: worked-problems

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

*本节的目标是把上一节的结论应用到一个非常实际的问题上：训练 LLaMA 3 系列（herd）模型。与前几节不同，我们希望你自己完成其中的大部分工作。因此，我们把每个小节的答案都隐藏了起来，方便你先自己尝试作答。试着拿起笔手算一下吧！*

### LLaMA 3 长什么样？ {#what-does-llama-3-look-like}

LLaMA-3 模型系列<d-cite key="llama3"></d-cite>包含 3 个主要模型：LLaMA 3 8B、70B 和 405B。我们主要关注 70B，而把 8B 和 405B 留给你在本节末尾的练习题中自行探索。下面是 LLaMA 3-70B 的架构，取自 LLaMA 的 [HuggingFace 页面](https://huggingface.co/NousResearch/Meta-Llama-3-70B/blob/main/config.json)。

| **超参数**                  | **取值**  |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 80        |
| $$d_\text{model}$$ (D)      | 8,192     |
| $$d_{ff}$$ (F)              | 28,672    |
| $$n_\text{heads}$$ (N)      | 64        |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

为了说明这些信息有多容易找到，下面就是配置文件本身，以及对应关系：

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

*把许多不同开源 LLM 的这些数字整理成一张大表会很有用，这样你就能快速比较它们各自做出的设计决策。*

### 计算参数量和 FLOPs {#counting-parameters-and-flops}

**问题：** 根据这张表，我们能算出 LLaMA 3-70B 的参数量吗？🤫 让我们运用[第 4 部分](../transformers)的内容，看看能不能得到 70B！

| 参数             | 公式                                                                                                                                              | 数量                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| FFW 参数         | d_model * d_ff * 3（分别对应 SwiGLU 的 gate、up 和 down 投影）* n_layers                                                                          | 8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9**                    |
| 词表参数         | 2（输入和输出嵌入）* n_embeddings * d_model                                                                                                       | 2 * 128,256 * 8,192 = **2.1e9**                              |
| 注意力参数       | n_layers * [ 2（对应 q 嵌入和拼接后的输出投影）* d_model * n_heads * d_qkv + 2（对应 k 和 v）* d_model * n_kv_heads * d_qkv]                      | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9** |
|                  |                                                                                                                                                   | 56.3e9 + 2.1e9 + 12e9 = **70.4e9**                           |

太好了！我们得到了预期的数字。你会注意到，正如预期的那样，FFW 参数在总参数量中占绝对主导，不过注意力部分也不可忽视。

<p markdown=1 class="takeaway">**要点**：MLP 块中的 3 个大权重矩阵比 Transformer 中所有其他数组都大得多，因此在推算模型内存或 FLOPs 时，我们通常几乎可以忽略其他所有参数。对 LLaMA 3-70B 来说，它们占了 70B 参数中的 56B。</p>

现在来看 FLOPs！*回忆一下[第 4 部分](../transformers)中关于训练的一般规则。*

**问题：** LLaMA-3 每个训练步对每个 token 要执行多少 FLOPs？*这有助于我们确定整个训练过程的开销有多大。*

{% details 想好之后，点击此处查看答案！ %}

**答案**：如[第 4 部分](../transformers)所示，每个 token 大约需要 $$6 \cdot \text{param count}$$ 次 FLOPs，所以这里大约是 `6 * 70e9 = 4.2e11` FLOPs / token。也就是说，每个训练步中每个 token 大约需要半个 TFLOP。假设我们是计算受限的，并且 FLOPs 利用率达到完美，那么在单颗 TPU v5p 芯片上这大约需要 `4.2e11 / 4.59E+14 = 1ms`。

{% enddetails %}

**问题：** LLaMA 3 大约训练了 15 万亿个 token。总共需要多少 FLOPs？

{% details 想好之后，点击此处查看答案！ %}

**答案**：这很简单，总量就是 `4.2e11 * 15e12 = 6.3e24 FLOPs`，即 6.3 yottaFLOPs。这可真不少！在单颗 TPU 上，这需要 `6.3e24 / 4.59E+14 = 435 years`。这也是很长的时间！

{% enddetails %}

**问题：** 假设我们想在一个完整的 TPU v5p Pod（16x20x28 = 8960 颗芯片）上训练。如果使用 bfloat16、MFU 为 40%，并假设是计算受限的，训练需要多长时间？

{% details 想好之后，点击此处查看答案！ %}

**答案**：我们知道每颗 TPU v5p 每秒可以执行 4.59e14 FLOPs。在 40% MFU 下，这大约需要 `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 seconds`。**大约是 44 天！** 只要我们真能达到 40% MFU，这个时间相当合理。

{% enddetails %}

**问题：** LLaMA 3-70B 预训练时使用的批大小（batch size）约为 4M 个 token。用这个批大小训练，我们至少需要多少颗 TPU？*你可以假设参数为 bfloat16、优化器状态为 float32，并且每层做 2 次激活值检查点（分别在层的开头和注意力之后）。*

{% details 想好之后，点击此处查看答案！ %}

**答案**：这个问题主要考察的是内存占用，因为这是对可用算力唯一的硬性约束。训练期间，HBM 主要有三种用途：模型参数、优化器状态和激活值检查点。如果假设权重为 bfloat16、优化器状态为 float32，并采用一种*非常*保守的激活值检查点方案（每层 2 次），我们有：

| **参数** | 2 * 70GB | ~140GB |
| **优化器状态** | 8 * 70GB | ~560GB |
| **激活值检查点** | 2 * 8192 * 4e6 * 2 * 80 | ~10.5TB |
| **总计**                 |                         | ~11.2TB |

这里的总量约为 11.2TB。你会注意到，即使采用非常保守的检查点方案，激活值检查点仍在内存占用中占据压倒性的主导地位。严格来说，我们可以改为每层 1 个检查点，或者使用微批次，但这已经是一个合理的估算了。在这些假设下，由于每颗 TPU v5p 有 96GB 的 HBM，我们需要 `11.2e12 / 96e9 = 117` 颗 TPU。其实并不算多！

*那为什么我们不这么做呢？* 因为这样训练需要 `44 days * 8960 / 117 = 3369 days`，差不多十年。**这太久了。** 不过，这清楚地表明：我们使用这些大规模集群，并不是因为受限于内存，而是因为我们需要额外的 FLOPs。此外，由于我们做检查点的频率很低，实际执行的 FLOPs 接近 8ND，而不是 6ND。
{% enddetails %}

**问题：** 在与上题相同的假设下，如果使用 8960 颗 TPU v5p 芯片，每颗芯片会占用多少内存？

{% details 想好之后，点击此处查看答案！ %}

**答案**：总内存仍然约为 11.2TB，因此每颗芯片大约只占用 1.3GB，基本上可以忽略不计。即使采用激进得多的检查点方案，例如每层 12 个检查点，每颗芯片也只需要 8GB。在这种规模下训练，我们离内存受限还远得很。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：从技术上讲，即使是非常大的模型也可以在非常小的拓扑上训练，只不过可能需要很长时间。只要能计算出一次训练运行的总 FLOPs，再假设一个适中的 MFU 和已知的拓扑，就能大致估算出训练时间。</p>

### 如何为训练分片 LLaMA 3-70B {#how-to-shard-llama-3-70b-for-training}

沿用上面的设定：假设我们想在一个 8960 颗芯片的 TPU v5p Pod 上，用 4M token 的批大小（每批 1024 条长度为 4096 的序列）训练 LLaMA 3-70B。我们来讨论一下这个模型的最佳分片策略是什么。

**问题：** 在上述假设下，我们能只用 FSDP 来训练模型吗？首先，假设我们不能做任何序列并行/上下文并行。*这应该是你首先想到的方案，因为它很简单，而且如果行得通的话，不会引入任何额外的通信。*

{% details 想好之后，点击此处查看答案！ %}

**答案**：这个答案会有点咬文嚼字。如上所述，LLaMA 3-70B 最初使用长度为 4K 的序列训练，因此 4M token 的批大小对应的*序列批大小*为 1024。这意味着我们实际上最多只能在 1024 颗芯片上做纯数据并行/FSDP，*因为我们只有这么多条序列可以用来做数据并行*。所以，按“完全数据并行、没有额外通信”这一简单含义来说，答案是否定的。下一个问题会回答一个稍微不那么咬文嚼字的版本。

{% enddetails %}

**问题：** 现在放宽“不做任何序列分片”的要求。如果允许同时在批次轴*和*序列轴上做 FSDP，我们能在 8960 颗芯片上只用 FSDP 训练 LLaMA 3-70B 吗？

{% details 想好之后，点击此处查看答案！ %}

**答案**：既然现在也允许做序列并行/上下文并行，我们就能扩展到大得多的规模。首先计算每个设备的批大小。如果做 8960 路 FSDP，每颗 TPU 的批大小就是 `4 * 1024 * 1024 / 8960 = 468 tokens`。从上一节我们知道，当 $$\text{per device batch size} < 2550 / M_X$$ 时，FSDP 会变成 ICI 受限。由于在完整的 3D Pod 上我们可以把 3 个轴都用于此，这会给出 850 的下界，而我们远低于这个值。**所以答案是否定的，即使用上 3 个轴也不行。我们将完全处于通信受限状态。**

{% enddetails %}

**问题：** 现在来看张量并行与 FSDP 混合使用的情况。是否存在某种组合能让我们保持计算受限？如果存在，FSDP 和张量并行各应该做多少？

{% details 想好之后，点击此处查看答案！ %}

**答案**：首先检查一下这是否可行。我们知道，如果每颗芯片的批大小小于 $2550^2 / 2F = 113$，就会通信受限。如上所示，我们略高于这个值。太好了！接下来，为了选出最优的 FSDP 规模，我们可以使用公式

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

取一个合理的 2 的倍数，大致就是 2048 路 FSDP 和 4 路张量并行。这样应该效果很好！

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：我们可以在一个完整的 TPU v5p Pod 上，用 4M token 的批大小训练 LLaMA-3，混合使用数据并行（1024 路）、序列并行（2 路）和张量并行（4 路），而不会通信受限。如果尝试纯 FSDP 或 FSDP + 序列并行，就会通信受限。我们在上一节推导出的这些公式非常实用。</p>

## 练习题 {#worked-problems}

**问题 1 [把 LLaMA 70B 扩展到更多芯片]：** 假设我们想用相同的批大小在 4 个 Pod 上训练 LLaMA 3-70B。我们会采用什么并行方案？会是计算受限还是通信受限？训练大约需要多长时间？*注意要使用正确的 Roofline 上限。*

**问题 2 [LLaMA 405B]：**

(a) 使用 LLaMA 3-405B 的[配置](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B/blob/main/config.json)（这是一个需要申请访问权限的模型，因此你可能需要登录并申请访问才能查看），像上面那样列出一张包含所有关键超参数的表。这个模型总共有多少参数？每个训练步需要多少 FLOPs？如果训练 15T 个 token，总共要执行多少 FLOPs？

(b) 假设我们想在 8 个 TPU v5p Pod 上训练。我们会采用什么并行方案？训练需要多长时间？会是计算受限还是通信受限？

<h3 markdown=1 class="next-section">第 6 部分到此结束。关于 Transformer 推理的第 7 部分，请点击[这里](../inference)。</h3>
