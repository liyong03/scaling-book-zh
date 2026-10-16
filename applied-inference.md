---
layout: distill
title: "在 TPU 上部署 LLaMA 3-70B"
# permalink: /main/
description: "让我们仔细看看如何在 TPU v5e 上部署 LLaMA 3-70B 模型。在 Roofline 意义下，部署不同模型的成本有多高？它们的 KV 缓存有多大？我们应该用多大的批大小？推理时参数和激活值是如何分片的？让我们对生产环境中的延迟和吞吐量做一些粗略估算。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "../inference"
previous_section_name: "第 7 部分：推理"

next_section_url: ../profiling
next_section_name: "第 9 部分：性能分析"

giscus_comments: false
giscus_discussion: 10

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
  - name: "LLaMA 的部署情况如何？"
    id: whats-the-llama-serving-story
  - subsections:
    - name: "思考吞吐量"
      id: thinking-about-throughput
    - name: "预填充呢？"
      id: what-about-prefill
  - name: "可视化延迟与吞吐量的权衡"
    id: visualizing-the-latency-throughput-tradeoff
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

*本节将探讨部署 LLaMA-3 需要什么，以及能做到多高效。和前面的“应用”章节一样，在查看答案之前，请先试着用纸笔自己算一算！*

## LLaMA 的部署情况如何？ {#whats-the-llama-serving-story}

我们先回顾一下 LLaMA 3-70B 的结构（参见[第 6 部分](../applied-training)）：

| **超参数**                  | **取值** |
| --------------------------- | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{KV 头数}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

从一个简单的问题开始：**我们应该用什么硬件来部署？** 答案基本上是：哪种硬件的 FLOPs / 美元最便宜就用哪种。<d-footnote>这并不总是成立，有时更关键的是更大的 HBM 或更高的 ICI 带宽，而不是 FLOPs，但这是一个不错的经验法则。</d-footnote> 因此，我们通常希望在 TPU v5e 上部署，它是我们目前专用的推理芯片（成本数据来自 2025 年 2 月的 [Google Cloud 定价](https://cloud.google.com/tpu/pricing)）：

| **TPU 类型** | **bfloat16 FLOPs/s** | **Google Cloud 美元 / 小时** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

每块 TPU v5e 有 16GB 的 HBM，这要求我们相当激进地对模型进行分片。我们先来想想一些可能对我们重要的基本量：

**问题：** LLaMA 3-70B 每个 token 的 KV 缓存有多大？*可以假设我们以 int8 存储它们。这决定了在给定拓扑上我们的批大小能有多大。*

{% details 想清楚之后再点击这里！ %}

LLaMA 3-70B 有 8 个 KV 头，所以每个 token 的大小是 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`。

**注意这有多大！** 如果序列长度是 32k token（这很常见），那么会占用 `160e3 * 32,768 = 5.3GB / sequence`。当 BS=240 时，这就是 1.3TB！由于每块 TPU v5e 只有 16GB，我们大约需要 `(70e9 + 1.3e12) / 16e9 = 86` 块 TPU v5e 芯片才能装下这么多内存。还要注意，与 70GB 的模型参数相比，这个量有多大。

{% enddetails %}

**问题：** 假设我们想以批大小 32、序列长度 8192 部署 L3 70B，所有内容（参数和 KV）都用 int8。总共会占用多少内存？能部署它的最小切片是多大？

{% details 答案 %}

由于 int8 下每个 token 的 KV 是 `160e3` 字节，总 KV 内存为 `160e3 * 8192 * 32 = 41.9e9` 字节。参数是 `70e9` 字节，因为每个参数占 1 字节。因此，总内存占用是 `41.9e9 + 70e9 = 112GB`。

我们能用的最小切片需要 `112e9 / 16e9 = 7` 块 TPU，或者（取整到偶数大小）TPU v5e `4x2`。这会装得非常紧，考虑到其他开销后可能不太装得下，所以我们至少可能需要一个 `4x4`（或者降低批大小）。

{% enddetails %}

**问题：** 在这个批大小和量化设置下，在 TPU v5e `4x2` 上，每个解码步的延迟大约是多少？吞吐量（tokens / sec / chip）是多少？`4x4` 呢？*假设我们以 bfloat16 执行 FLOPs，并且所有东西都已完全分片。*

{% details 答案 %}

我们可以借用上一节的公式：

$$\begin{align*}
\tiny \text{理论步时间（一般情况）} = \underbrace{\frac{\text{批大小} \times \text{KV 缓存大小}}{\tiny \text{总内存带宽}}}_{\text{注意力（始终带宽受限）}} + \underbrace{\max\left(\frac{2 \times \text{批大小} \times \text{参数量}}{\text{总 FLOPs/s}}, \frac{\text{参数大小}}{\text{总内存带宽}}\right)}_{\tiny \text{MLP（可能计算受限）}}
\end{align*}$$

这里我们的临界批大小约为 120，因为参数是 int8 而 FLOPs 是 bfloat16。我们也可以手动计算右侧的最大值，但这基本上是我们已经做过好几次的计算了。**所以无论是矩阵乘法还是 FLOPs，我们都深处内存受限区间。**

那么严格只看内存带宽，步时间基本上就是 `(KV size + param size) / (8 * HBM bandwidth) = 112e9 / (8 * 8.2e11) = 17ms`。**所以理论上步时间约为 17ms。** 吞吐量为 `32 / .017 = 1882 tokens / sec`，即 `1882 / 8 = 235 tokens / sec / chip`。

这里有一点需要注意：要检查矩阵乘法是否会受 ICI 限制。这里我们可以为它分配 2 个轴，所以理论上当 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$ 时才会受 ICI 限制，因此我们完全没问题！

如果在 `4x4` 上运行，ICI 方面依然没问题，所以延迟会降到 `17 / 2 = 8.5ms`，但每芯片吞吐量保持不变。

{% enddetails %}

### 思考吞吐量 {#thinking-about-throughput}

我们花点时间单纯地思考一下吞吐量。当我们为吞吐量做优化时，我们希望处于计算受限状态，也就是接近用满 TPU MXU 的全部算力。通常这意味着我们希望批大小尽可能大，从而做尽可能多的工作。

**问题：** 在 TPU v5e 上，如果权重和激活值都用 bfloat16，批大小需要多大才能让矩阵乘法计算受限？如果权重用 int8 但 FLOPs 用 bfloat16 呢？如果权重是 int8、FLOPs 也是 int8 呢？

{% details 答案 %}

正如第 7 部分所讨论的，对于任何满足 $B \ll D, F$ 的 bfloat16 矩阵乘法，我们有

$$\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM 带宽}} = 240
\end{equation*}$$

当权重是 int8 时，分母少了一个因子 2，于是得到 $2BDF / DF = 2B > 240$，也就是 $B > 120$，是之前临界批大小的一半。这对我们很有帮助！当权重和 FLOPs 都是 int8 时，我们必须使用 int8 对应的 TPU FLOPs/s，它从 bfloat16 的 1.97e14 变为 3.94e14，几乎翻倍。这意味着我们又回到了起点，大约 $B > 240$。

int8 权重加 bfloat16 FLOPs 的情形相当常见，因为无损地量化参数往往比做低精度运算更容易。

{% enddetails %}

**问题：** 在 8k 上下文下，分别使用 bfloat16、int8 和 int4（KV 和参数都如此），我们能部署 LLaMA 3-70B 的最小 TPU v5e 拓扑是多大？*这道题可以把 KV 缓存视为小到可以忽略。*

{% details 答案 %}

这很简单！如果我们能接受极小的批大小，那么唯一的限制就是把参数内存装进 HBM，即 `ceil(num_params * sizeof(dtype) / HBM per TPU)`，或者 `ceil(70e9 * sizeof(dtype) / 16e9)`，再取整到最接近的合理拓扑（2 的某个倍数）：

| dtype | 参数大小 | 每 token 的 KV 大小（字节） | 最少 TPU v5e 数 | 实际最小切片 | 剩余可用于 KV 缓存的 HBM | 8k 下的 KV 缓存数量 |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 块芯片  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 块芯片   |             58              |         43         |
| int4  |    35GB    |          81kB           |     2.81     |  2x2 = 4 块芯片   |             29              |         43         |

这相当酷！它告诉我们，只要愿意，就可以把 LLaMA 70B 装进一个 TPU v5e 2x2。不过你会注意到 KV 缓存的数量非常少。这就是我们的批大小！这意味着我们的 FLOPs 利用率会非常糟糕。我们会很乐意使用更大的拓扑，以便把批大小推到 240。

{% enddetails %}

**问题：** 假设我们使用这些拓扑上能装下的最大批大小，每个生成步的延迟大约是多少？

{% details 答案 %}

这也很简单，因为我们选择的批大小恰好填满了全部 HBM！问题只是把一整块 TPU v5e 的字节加载进 MXU 需要多久。这就是 `v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`，所以是 **19ms / 步**。假设生成长度的中位数是 512 个 token，那么每次解码大约需要 9s。注意，用更小的批大小可以获得略好一点的延迟，例如如果只看 int4 的模型参数，最小延迟约为 10ms / 步，因为此时 HBM 不再是满的。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：我们总可以通过计算“把模型的所有参数从 HBM 加载进 MXU 需要多久”来给出解码延迟的下界。当 KV 缓存很小时，可以把每一层看作只是逐块加载权重然后丢弃。除非我们使用很大的批大小或大量设备间通信，这通常是一个合理的界（误差在 1.5 倍以内）。当批大小更大时，我们还需要对 KV 缓存的加载建模，因为它会超过参数成为主导。</p>

同样，在 FLOPs 受限区间（例如训练或大批量推理），我们可以使用 $$\text{总 FLOPs} / (N \cdot C) = 2 \cdot \text{参数量} \cdot B / (N \cdot C)$$ 这个下界，它假设没有任何通信。

**问题：** 对于上面每种情况，每芯片的吞吐量（以“查询数 / 芯片”计）是多少？*可以假设解码长度的中位数是 512 个 token。*

{% details 答案 %}

这是一个重要的问题，因为它与每 token 成本严格对应。

在关于解码长度中位数的假设下，吞吐量就是 $$B / (\text{每步延迟} \cdot \text{中位步数} \cdot N) \approx 43 / (0.019 * 512 * N)$$。这大约给出 $$(4.42 / N)$$ QPS，代入 $$N$$ 得到：

|  dtype   | QPS / 芯片 |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.55    |
|   int4   |    1.11    |

注意这相当乐观，因为它完全忽略了前向传播的工作内存（分配给激活值和注意力的内存）。在使用 Flash Attention 的情况下这并不荒谬，但也不现实。真实数字很可能只有这个的 1/2 左右。若要追求绝对最大的吞吐量，我们可能会希望把芯片数量增加到两倍以上，同时也大幅增加批大小。

{% enddetails %}

**问题：** 如果把上面每个例子的拓扑都扩大一倍，峰值吞吐量会如何变化？

{% details 答案 %}

如果在 bfloat16 下使用 4x8 切片，我们会有 372GB 剩余可用于 KV 缓存，这让我们能把批大小提高到 140。由于步时间保持不变，吞吐量将是 `14.39 / num_chips`，即

|       dtype       | QPS / 芯片 |
| :---------------: | :--------: |
| bfloat16（4x8 上） |    0.44    |
|   int8（4x4 上）   |    0.90    |
|   int4（2x4 上）   |    1.80    |

继续增大还会带来更大的收益！主要结论是：如果受限于 KV 缓存大小，那么**最小的拓扑并不总是性能最好的拓扑**。

{% enddetails %}

**问题：** 现在我们来深入探讨分片问题。假设我们想在 TPU v5e 4x8 上以 bfloat16 部署。在 TPU v5e 4x8 上生成时，我们应该为模型采用什么分片方式？能否避免通信受限？

{% details 答案 %}

正如上一节所讨论的，生成阶段我们实际上只有一种分片选择：模型并行。在变成通信受限之前，我们能做到多大程度的模型并行？正如上一节讨论过的，大致在以下情况下模型会变成通信受限：

$$Y > \frac{F \cdot M_Y}{2200}$$

对于 LLaMA 3-70B，`F = 28,672`，所以如果在 2 个轴上做模型分片，大约得到 $$Y = 28672 \cdot 2 / 2200 = 26$$，因此一般来说我们可以扩展到大约 16 块芯片而不会通信受限，这允许我们使用 `4x4`，但不能使用 `4x8`。一般而言，由于计算与通信无法完美重叠，即便这个估计也过于乐观。

**要点：我们实际上无法在 4x8 上用纯模型并行来部署。** 这里我们最多能用 4x2，*或许*能用 4x4。

不过，正如我们讨论过的，当批大小较小时，我们通常可以做更多的模型并行而不会明显损害吞吐量，因为此时模型受内存带宽限制而不是 FLOPs 限制。我们之前说过这个值大约是 $Y=F / (8\cdot B)$，所以如果批大小为 64，理论上我们可以把模型并行度提高到 `Y = 28,672 / (8 * 64) = 56` 路，才会变成 ICI 受限。为了验证这一点，我们可以看单个矩阵乘法的 $T_\text{ICI 通信}$、$T_\text{HBM 通信}$ 和 $T_\text{math}$。显然有：

$$\begin{align*}T_\text{ICI 通信} = \frac{2BD}{W_\text{ici}} && T_\text{HBM 通信} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

对于 `4x8`，可以得到 $T_\text{ICI 通信}$ = `(2 * 64 * 8192) / 9e10 = 11us`，$T_\text{HBM 通信}$ = `(2 * 8192 * 28,672) / (32 * 8.2e11) = 18us`，以及 $T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`，所以理论上我们仍然受 HBM 带宽限制，这很好！*注意，从 `4x4` 扩展到 `4x8` 从吞吐量的角度来看可能没什么帮助，但它会降低延迟！*

如果看 int8 和 int4 的配置，我们*确实*可以用纯模型并行来实现。所以我们到了这样一个点：量化真正带来了 FLOPs 更快之外的实质性优势——它让我们在变成通信受限之前可以使用更大的批大小。**所以这个故事的结论是：我们无法在 4x8 上达到峰值吞吐量，但对于 int8 和 int4 配置，我们可以使用纯模型并行。**

{% enddetails %}

<p markdown=1 class="takeaway">**提示**：有用的模型并行的最大程度取决于 $$d_{ff}$$ 以及你在多少个轴上对模型分片。根据模型大小，这个最大值通常在 8 到 32 之间。你可以超出这一限制来改善延迟，但要付出一些吞吐量的代价。</p>

### 预填充呢？ {#what-about-prefill}

这里我们基本忽略了预填充，因为它要简单得多。让我们把几个概念结合起来，思考一下端到端的全貌。

**问题：** 假设预填充期间我们达到 40% 的 FLOPs 利用率。在 16 块 TPU v5e 芯片上，一次长度为 8192 的预填充需要多长时间？

{% details 答案 %}

在 8k token 下，我们稳稳处于计算受限，所以只需要考虑 FLOPs。我们知道模型有 `70e9` 个参数，所以每次前向传播使用 `2 * 70e9 * B` FLOPs。假设 MFU（FLOPs 利用率）为 40%，运行时间约为 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s`。与我们之前看到的数字相比，这实际上相当长！

{% enddetails %}

**问题：** 假设预填充长度的中位数是 8192 个 token，解码长度的中位数是 4096 个 token。假设生成批大小为 32。平均每步有多少序列完成解码？平均每步有多少 token 从 KV 缓存中被逐出？

{% details 答案 %}

这相当直接。由于解码长度的中位数是 4096 个 token，每个序列大约每 4096 个 token 结束一次（即每步约有 1 / 4096 的序列完成）。在批大小为 32 的情况下，这意味着每步有 `32 / 4096` 个序列被逐出。由于 KV 缓存长度大约是 `8192 + 4096`，每步被逐出的 token 数为 `32 * (8192 + 4096) / 4096 = 96`。通用公式是 $B * (P + G) / G$，其中 $P$ 和 $G$ 分别是预填充长度和生成长度。

{% enddetails %}

**问题：** 假设我们采用分离式部署（disaggregated serving），预填充长度的中位数是 8192，解码长度的中位数是 512。假设预填充和生成的延迟采用上面在 bfloat16 下计算的数值。预填充服务器与生成服务器的比例应该是多少，才能让两者都保持满载？

{% details 答案 %}

这是个挺有趣的问题。设 $P$ 为预填充服务器的数量，$G$ 为生成服务器的数量。一般来说，这是一个流水线问题：我们以 `P / prefill_latency` 的速率送入序列，并以 `B * G / (generate_latency * median_decode_length)` 的速率消耗它们。我们之前算出每个预填充步需要 `910ms`，在批大小 43（就当作 32 吧）时每个解码步需要 `19ms`。因此我们需要 `P / 0.91 = 32 * G / (0.019 * 512)`，即 `P = 3G`，也就是说我们需要的预填充服务器大约是生成服务器的 3 倍！

{% enddetails %}

## 可视化延迟与吞吐量的权衡 {#visualizing-the-latency-throughput-tradeoff}

继续以 LLaMA 70B 为例，我们来实际看看生成阶段不同批大小下的延迟和吞吐量。正如上一节对 PaLM 模型所展示的，这会给出一条吞吐量/延迟的 Pareto 前沿。假设采用 16 路张量并行，因为这是在 MLP 块中保持计算受限时可用并行度的一个合理上限。这里我们使用 TPU v5e 4x4 拓扑。**滑块控制序列长度，你可以看到更大的 KV 缓存带来的影响。**

<figure class="plotly-embed">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" title="LLaMA 3-70B 的延迟/吞吐量 Pareto 前沿" loading="lazy" scrolling="no"></iframe>
  <figcaption><b>图：</b>LLaMA 3-70B（int8 权重，16 路张量并行）在 TPU v5e 4x4 上的每芯片吞吐量与每 token 延迟的关系，每条曲线沿批大小扫描。拖动滑块可改变上下文长度；每条曲线止于 KV 缓存仍能装进 HBM 的最大批大小。</figcaption>
</figure>

* **看看成本与延迟之间的权衡有多么剧烈。** 以每 token 延迟翻倍为代价，我们可以把每 token 成本降低约 100 倍。此外，我们的延迟可以从小批大小时的 5.5ms 一直变化到超大批量时的 20 ms。
* 注意在 2k 上下文下，当达到 BS 120 的 Roofline 时（这里是 120，因为我们用 int8 权重但 bf16 FLOPs），吞吐量实际上会在约 1 token / ms / chip 处趋于平稳。然而随着序列长度增加，我们无法再把这个批大小装进内存，因此永远达不到完全饱和的点。
* 注意在相同吞吐量下，大批大小时的延迟要高得多，因为此时 KV 加载（而不是参数加载）成为主导。

我们可以把成本和延迟的来源拆分为参数加载时间、KV 加载时间和 FLOPs 时间，从而更好地理解这一点。阴影区域是我们预期 MLP 块处于计算受限的区间。

<figure class="plotly-embed">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" title="LLaMA 3-70B 的解码步时间分解" loading="lazy" scrolling="no"></iframe>
  <figcaption><b>图：</b>LLaMA 3-70B 在 TPU v5e 4x4 上每个解码步的时间随批大小的变化，拆分为参数加载、KV 缓存加载和 FLOPs 三部分。总时间等于 KV 加载时间加上参数加载与 FLOPs 两者中较大的那个。拖动滑块可改变上下文长度。</figcaption>
</figure>

这张图很能说明问题。可以看到，起初参数加载占据了延迟的绝大部分，直到批大小足够大，FLOPs 和 KV 加载才变得更显著。值得注意的是，在所有大于 2048 的序列长度下，我们花在 KV 缓存加载上的时间都比花在 FLOPs 上的多！**所以，虽然我们可以通过增大批大小来提高硬件利用率，但在长上下文下，KV 加载始终主导总步时间。**

<p markdown=1 class="takeaway">**要点：** 对于 LLaMA 3-70B，在几乎所有这些配置中，我们都严重受限于 KV 缓存的内存带宽（也受 HBM 限制），这凸显了减小 KV 缓存大小对生成吞吐量有多么重要。还要注意，这里的延迟/吞吐量权衡依然非常剧烈。</p>

{% details 这部分的代码相当简单。 %}

下面是计算这些 Roofline 的代码：

```py
import numpy as np

num_chips = 16  # 我们把总的模型并行度固定为 16
bytes_per_param = 1  # int8 意味着每个参数占 1 字节
param_count = 70e9
param_size = bytes_per_param * param_count
sequence_length = 8192  # 可以调整这个值

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80

def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(
    num_chips: int,
    sequence_length: int,
    param_size: float,
) -> int:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  required_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(required_chips <= num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(
    num_chips=num_chips,
    sequence_length=sequence_length,
    param_size=param_size,
)  # 求出能放得下的最大批大小
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_size * batch_sizes / (num_chips * flops)  # 在 2ND 的意义上大致成立

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # 生成阶段始终是带宽受限的

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

注意我们是如何非常明确地把延迟拆分为两个来源的：KV 加载和参数加载；以及延迟是如何受 FLOPs 或通信限制的——取两者中较大的那个。

{% enddetails %}

## 练习题 {#worked-problems}

下面是几道练习题。其中一些重复了上文已经解答过的内容，但可能在教学上仍有帮助。

**问题 1：** LLaMA 3-405B 的每次前向传播每个 token 使用多少 FLOPs？假设我们受 FLOPs 限制，在 TPU v5e 的 N 块芯片上，单次前向传播的下界是多少？如果受通信限制呢？*忽略模型无法装进单块芯片这一事实。*

**问题 2：** 假设我们想用 int8 权重和 int8 KV 缓存、以 BS240 部署 LLaMA 3-8B。(a) 模型参数、(b) KV 缓存以及 (c) 峰值工作激活值（粗略估计）分别占用多少字节？能运行它的最小拓扑是多大？

**问题 3：** 你会如何在 TPU v5e 上部署 LLaMA 3-405B？假设使用 int8 权重和 bfloat16 FLOPs。假设我们有 15ms / token 的硬性限制，能实现的最高吞吐量配置是什么？理论上的最小步时间是多少？

**问题 4：** 学习 LLM 的最好方法就是从零实现一个。构建完整的训练流水线既麻烦又昂贵，但把你可以[从 HuggingFace 下载](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/)的训练好的权重变成一个可用的推理实现，是极具启发性的。大致来说，你应该尝试完成以下步骤：

1. 下载 LLaMA 3 8B 的权重并在 Colab 中加载。可视化这些权重（我喜欢用 [TreeScope](https://github.com/google-deepmind/treescope) 来做这件事），看看你能否识别出每个张量。数一数参数数量。和你预期的一致吗？MLP 和注意力中各有多少参数？

2. 实现模型完整的前向传播。你不需要操心预填充/解码或任何花哨的东西。只要做到能输入一个序列并得到合理的下一个 token 概率即可。你应该能够输入一个提示词，得到概率输出，选出概率最高的那个，然后重复。这种采样虽然慢，但能用。尽量只参考 LLaMA-3 论文，不过 [`jax-ml/jax-llm-examples/llama3`](https://github.com/jax-ml/jax-llm-examples/tree/main/llama3) 仓库可以作为细节上的良好参考（注意位置嵌入和因果掩码）。你的目标是让模型输出连贯的 token。*如果你能拿到超过 16GiB 的 HBM，就可以在单块 TPU 上运行这一切。*

3. 现在是时候让它变快了。实现 KV 缓存：把一次前向传播中的键/值激活值保存下来，并在后续的前向传播中对它们做注意力。这会让你的采样循环快得多。

4. 实现分离的预填充服务器和解码服务器。你可以把它们放在不同的芯片组上。预填充服务器一次只处理一个提示词，然后把它发送给一个批处理的解码服务器，由后者处理成批的 token。

5. 用 Pallas 实现 Flash Attention。这会让注意力计算高效得多。*我认为不借助参考实现去尝试完成这一步是很好的练习。*

<h3 markdown=1 class="next-section">第 8 部分到此结束！第 9 部分将深入介绍 XLA 和 TPU 性能分析，请点击[这里](../profiling)。</h3>
