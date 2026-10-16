---
layout: distill
title: "关于 Transformer 推理的一切"
# permalink: /main/
description: "在 Transformer 上做推理可能与训练截然不同。部分原因在于推理引入了一个需要考虑的新因素：延迟。在本节中，我们将从“从模型中采样单个新 token”一路讲到“作为推理引擎的一部分，在多个加速器切片上高效扩展大型 Transformer”。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: "../applied-training"
previous_section_name: "第 6 部分：训练 LLaMA"

next_section_url: ../applied-inference
next_section_name: "第 8 部分：部署 LLaMA"

bibliography: main.bib

giscus_comments: false
giscus_discussion: 9

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
  - name: "Transformer 推理基础"
    id: the-basics-of-transformer-inference
  - subsections:
    - name: "我们究竟想优化什么？"
      id: what-do-we-actually-want-to-optimize
    - name: "线性运算：瓶颈在哪里？"
      id: linear-operations-what-bottlenecks-us
    - name: "注意力呢？"
      id: what-about-attention
    - name: "LLM 延迟与吞吐量的理论估计"
      id: theoretical-estimates-for-llm-latency-and-throughput
    - name: "内存呢？"
      id: what-about-memory
    - name: "为 LLaMA 2-13B 建模吞吐量与延迟"
      id: modeling-throughput-and-latency-for-llama-2-13b
  - name: "提升生成吞吐量与降低延迟的技巧"
    id: tricks-for-improving-generation-throughput-and-latency
  - name: "在多个加速器上分布式推理"
    id: distributing-inference-over-multiple-accelerators
  - subsections:
    - name: "预填充"
      id: prefill
    - name: "生成"
      id: generation
    - name: "对 KV 缓存进行分片"
      id: sharding-the-kv-cache
  - name: "设计高效的推理引擎"
    id: designing-an-effective-inference-engine
  - subsections:
    - name: "连续批处理"
      id: continuous-batching
    - name: "前缀缓存"
      id: prefix-caching
    - name: "来看一个实现：JetStream"
      id: lets-look-at-an-implementation-jetstream
  - name: "练习题"
    id: worked-problems
  - name: "附录"
    id: appendix

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

## Transformer 推理基础 {#the-basics-of-transformer-inference}

假设你已经训练好了一个 Transformer，现在想用它来生成一些新的序列。*归根结底，基准分数上升、损失曲线下降，都只是代理指标——真正动起手来时，是否会发生有意思的事情才是关键！*<d-footnote>从历史上看，你可以在完全不碰推理的情况下对 Transformer 做大量研究——基于打分的多项选择基准测试，无需完善的 KV 缓存或生成循环实现就能高效运行。这意味着，尤其是在研究代码库中，推理代码路径上往往有很多唾手可得的优化空间。</d-footnote>

采样在概念上很简单。我们输入一个序列，心爱的 Transformer 就会吐出 $$\log p(\text{下一个 token}_i \vert \text{之前的 token})$$，即所有可能的下一个 token 的对数概率。我们可以从这个分布中采样，得到一个新的 token。把这个 token 追加到序列末尾并重复这个过程，就能得到一个 token 序列，作为提示词（prompt）的续写。

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>图：</b>Transformer 的朴素采样。蓝色的 logits 给出了下一个 token 的分布，我们可以从中采样。注意每一步都会重新处理整个前缀，导致该算法的运行时间为 $\Theta(n^2)$。" %}

我们刚才描述的是 Transformer 采样的朴素实现。它虽然能用，但**我们在实践中从不这样做**，因为每生成一个 token 都要重新处理整个序列。要生成 $$n$$ 个 token，这个算法在前馈层上是 $$O(n^2)$$，在注意力机制上是 $$O(n^3)$$！

**如何避免这种情况？** 与其每次都做完整的前向传播，事实证明我们可以保存每次前向传播的一些中间激活值，从而避免重新处理之前的 token。具体来说，由于在点积注意力中一个 token 只会关注它之前的 token，我们可以简单地把每个 token 的 key 和 value 投影写入一个名为 **KV 缓存（KV cache）** 的新数据结构中。一旦保存了过去 token 的 key/value 投影，后续 token 只需计算各自的 $$q_i \cdot k_j$$ 乘积，而无需在之前的 token 上执行任何新的 FLOPs。太棒了！

有了这一点，推理就分为两个关键部分：

* <b style="color: red;">预填充（prefill）</b>：给定一个长提示词，我们同时处理提示词中的所有 token，并把得到的激活值（具体来说是 key-value 投影）保存在 **“KV 缓存”** 中。我们还会保存最后一个 token 的 logits。
* <b style="color: blue;">生成（generation）</b>：给定 KV 缓存和之前的 logits，我们从 logits 中增量地采样一个 token，把该 token 送回 Transformer，并为下一步产生一组新的 logits。同时把这个新 token 的 KV 激活值追加到 KV 缓存中。重复这一过程，直到遇到特殊的 `<EOS>` token 或达到某个最大长度限制。

下面是使用 KV 缓存进行采样的示意图：

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>图：</b>使用 KV 缓存的高效 Transformer 采样示意图。<b style=\"color: red;\">预填充</b>处理我们的提示词，并把每个 token 的 key-value 激活值全部保存在缓存中。<b style=\"color: blue;\">生成</b>接收这个缓存（以及最后一个 token 的 logits），采样一个新 token，再让这个新 token 通过模型，对 KV 缓存做注意力计算，并把新 token 的 key-value 投影写回缓存。在 MLP 块中，这是一个 $O(n)$ 的算法。" %}

通过使用 KV 缓存进行采样，我们把生成 $n$ 个 token 的时间复杂度降低到了前馈层上的 $$O(n)$$ 和注意力上的 $$O(n^2)$$，因为我们不再重复处理之前的 token。不过，生成一个序列仍然需要多次前向传播——当你向 Gemini 或 ChatGPT 提问、结果以流式方式返回时，背后发生的正是这件事。每个 token（通常）都是对一个巨大模型的一次独立（但部分缓存）的 Transformer 调用。

我们很快就会看到，<b style="color: red;">预填充</b>和<b style="color: blue;">生成</b>是截然不同的两头猛兽——Transformer 推理其实是两个伪装成一个的任务！与训练相比，KV 缓存也是一个全新的、重要的复杂性来源。

### 我们究竟想优化什么？ {#what-do-we-actually-want-to-optimize}

在继续深入之前，值得强调推理中一个全新的方面：延迟。训练时我们只关心吞吐量（**每个芯片**每秒处理的 token 总数），而推理时我们还必须关心生成 token 的速度（包括**首 token 时间（Time To First Token，TTFT）**和**每 token 延迟**）。例如：

* 用于评测和数据生成的**离线批量推理**只关心推理的总体成本，并不在意单个样本的延迟。
* **聊天界面/流式任务**需要在大规模下低成本运行，同时保持较低的 TTFT，并且生成 token 的速度要超过人类的阅读速度。
* **边缘推理**（例如在你的笔记本电脑上运行 `llama.cpp`）一次只需服务一个用户，要求尽可能低的延迟，且可能受到严格的硬件限制。

最大化硬件利用率仍然至关重要，它有助于降低成本和 TTFT；但与训练不同，它在所有场景下并不*一定*能转化为单个用户更好的体验。加速器、系统和模型架构层面的许多优化，都是在延迟、吞吐量、上下文长度乃至模型质量之间进行权衡。

### 更细粒度地看 Transformer {#a-more-granular-view-of-the-transformer}

到目前为止，我们大多把 Transformer 看作一堆前馈块的堆叠。虽然从 FLOPs 和内存的角度看这通常是合理的，但它不足以准确地对推理建模。<d-footnote>在本节中你会注意到，推理远不如训练那样宽容。我们通常拥有的 FLOPs 少得多，批处理的机会更少，对延迟也敏感得多。KV 缓存也让推理变得复杂得多。</d-footnote>正如我们在[第 4 部分](../transformers)中看到的，Transformer 前向传播的主要组成部分是：

1. **一系列线性运算**，包括 MLP（$W_{in}$、$W_{out}$）以及注意力的 QKV 投影和输出投影（$W_Q$、$W_K$、$W_V$ 和 $W_O$）。它们都需要从 HBM 读取参数和一批激活值，执行一些 FLOPs，再把结果写回 HBM。
2. **点积注意力**。我们需要从 HBM 读取一批 key-value 投影和一批 query 激活值，做一些内积和 softmax 运算，再把注意力结果写回 HBM。
3. **其他所有部分**，包括应用层归一化（layer norm）、激活函数、token 采样、更新 KV 缓存以及位置嵌入。这些确实也需要一些 FLOPs，但开销被上述运算所主导，或者被融合进上述运算中。

在接下来的几个小节中，我们将在预填充和生成的背景下分别考察上述各部分，并探讨什么最可能成为性能瓶颈。在单个加速器内部，我们是计算受限还是内存受限？我们想强调的是，对于预填充和生成，答案会有多么不同。

### 线性运算：瓶颈在哪里？ {#linear-operations-what-bottlenecks-us}

无论位于 MLP 块还是注意力中，我们的所有线性运算在概念上都是一样的。它们的算术强度取决于批大小。我们在[第 1 节](../roofline)中做过这个推导，但值得再重复一遍。来看一个 $\text{bf16[B, D]}$ 批次与一个 $\text{bf16[D, F]}$ 矩阵之间的单次矩阵乘法。它可以是大的 MLP 块（$W_\text{in}$ 或 $W_\text{out}$），也可以是较小的注意力投影之一（$W_Q$、$W_K$、$W_V$、$W_O$）。为了完成这次矩阵乘法，我们需要把这两个数组从 HBM 加载到 MXU 中，执行乘法，再把结果写回 HBM。和之前一样，我们有：

$$T_\text{math} = \frac{\text{计算 FLOPs}}{\text{加速器 FLOPs/s}} = \frac{2BDF}{\text{加速器 FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{通信字节数}}{\text{带宽 Bytes/s}} = \frac{2BD + 2FD + 2BF}{\text{带宽 Bytes/s}}$$

TPU 或 GPU 可以一边计算一边加载，从而让二者重叠，因此要做到计算受限，我们需要 $$T_\text{math} \geq T_\text{comms}$$，即：

$$\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{加速器 FLOPs/s}}{\text{带宽 Bytes/s}} \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} = 240$$

其中右侧是硬件的算术强度。现在假设 $D$ 和 $F$ 相对于 $B$ 非常大（通常我们的批次最多为 500，而 $D$ 和 $F > 10k$），就可以利用 $\small{2BD + 2DF + 2BF \approx 2DF}$ 来简化分母，得到

$$\begin{align*}
\frac{2BDF}{2BD + 2DF + 2BF} \approx \frac{2BDF}{2DF} \geq \frac{\text{加速器 FLOPs/s}}{\text{带宽 Bytes/s}} \\
\underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}}
\end{align*}$$

如果我们对权重进行量化，或在矩阵乘法中使用更低精度的 FLOPs，这个临界批大小就会改变。例如，如果把权重量化为 int8 或 fp8，$B_\text{crit}$ 会降低 2 倍。如果用 int8 或 fp8 进行 FLOPs 计算，$B_\text{crit}$ 会增大 2 倍。因此，如果令 $\beta = \text{每个参数的位数} / \text{每个激活值的位数}$，$\alpha_\text{hbm} = C / W_\text{hbm}$，那么临界批大小实际上是 $B_\text{crit} = \beta \alpha_\text{hbm}$。

<p markdown=1 class="takeaway">**要点：** 当且仅当（*iff*）每个副本的 **token** 批大小大于 $B_\text{crit} = C / W_\text{hbm} \cdot (\text{每个参数的位数} / \text{每个激活值的位数}) = \beta \cdot \alpha_\text{hbm}$ 时，Transformer 的矩阵乘法才是计算受限的。对于 TPU v5e 上的 bf16 激活值，这个值是 240 个 token。对于 H100，大约是 280 个 token。</p>

训练期间，所有矩阵乘法都会有很高的算术强度，因为我们在一个非常大的批次上复用同一份权重。**这种高算术强度同样适用于预填充，因为用户的提示词通常有数百甚至数千个 token 长。** 正如之前所见，TPUv5e 的硬件算术强度是 240，因此如果把一个长于 240 个 token 的序列输入到以 bf16 运行在该硬件上的稠密模型中，我们预计会是计算受限的，一切都没问题。比这更短的提示词在技术上可以批处理在一起以获得更高的利用率，但通常没有必要。

<p markdown=1 class="takeaway">**要点：** 在预填充期间，所有矩阵乘法基本上总是计算受限的。因此，只需最大化硬件利用率或 MFU（模型 FLOPs 利用率，Model FLOPs Utilization），就足以最大化每芯片吞吐量（即成本）并优化延迟（体现为 TTFT）。除非提示词极短，否则在单个提示词层面做批处理只会增加延迟，而对预填充吞吐量的提升很小。</p>

然而在生成期间，对于每个请求，我们一次只能对一个 token 做前向传播，因为步骤之间存在顺序依赖！因此，我们只能通过把多个请求批处理在一起、在批次维度上并行，来（轻松地）获得良好的利用率。我们稍后会详细讨论这一点，但实际上要在不影响延迟的情况下把许多并发请求批处理在一起是很难的。正因如此，**生成阶段要让硬件 FLOPs 饱和要困难得多。**

<p markdown=1 class="takeaway">**要点：** 在生成期间，总 token 批大小必须大于 $B_{\text{crit}}$，线性/前馈运算才能做到计算受限（在 TPU v5e 上对 bf16 参数而言是 240）。由于生成是逐 token 串行进行的，这就要求我们把多个请求批处理在一起，而这很难！</p>

*值得注意的是这个数字有多大！* 生成批大小为 240 意味着有 240 个并发请求同时在生成，对于稠密模型来说还意味着 240 个独立的 KV 缓存。这意味着在实践中很难做到，除非是某些批量推理场景。相比之下，在预填充期间一次处理超过 240 个 token 是很常规的事情，不过随着稀疏度的增加需要多加注意。

**注意，这个确切的数字会因量化类型和硬件而异。** 加速器在较低精度下通常能提供更多的算力。例如，如果参数是 int8 但计算用 bf16，临界批大小会降到 120。如果激活值和参数都是 int8，它又会跳回 240，因为 TPUv5e 可以提供 400 TOPs/s 的 int8 x int8 算力。

### 注意力呢？ {#what-about-attention}

当我们考察点积注意力运算时，事情会变得更复杂，尤其是因为我们必须把 KV 缓存考虑进来。我们只看纯多头注意力中的一个注意力头。在单个 Flash Attention 融合中，我们<d-footnote>这里我们做了相当大的简化，忽略了应用 softmax、掩码等操作中的非矩阵乘法 FLOPs。它们应当与计算或 HBM 读取重叠，但在某些 TPU 代际上这并不容易做到。虽然这些细节不会改变主要结论（即 KV 缓存通常是内存受限的），但它们值得留意。</d-footnote>：

1. 从 HBM 读取形状为 $\text{bf16[B, T, D]}$ 的 $Q$ 激活值。
2. 从 HBM 读取 $KV$ 缓存，它是一对 $\text{bf16[B, S, D]}$ 张量。
3. 在 $$QK$$ 矩阵乘法中执行 $2BSTD$ 次 FLOPs。借助 Flash Attention，我们不需要把 $\text{bf16[B, S, T]}$ 注意力矩阵写回 HBM。
4. 在注意力的 $$AV$$ 矩阵乘法中执行 $2BSTD$ 次 FLOPs。
5. 把得到的 $\text{bf16[B, T, D]}$ 张量写回 HBM。

综合起来，我们得到：

$$\text{多头注意力算术强度} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

对于预填充，由于我们做的是自注意力，$S=T$，所以上式简化为 $T^2 / 2T = T / 2$。这很好，因为它意味着**预填充期间注意力的算术强度为 $\Theta(T)$**。也就是说，注意力很容易做到计算受限。只要序列长度足够大，就没问题！

但由于生成阶段的序列维度微不足道，且 $B$ 和 $D$ 维度会相互抵消，我们可以做如下近似：

$$S \gg T = 1 \implies \frac{ST}{S+T} \approx 1$$

这很糟糕，因为这意味着我们无法做任何事情来提高生成期间注意力的算术强度。我们在加载一个庞大的 KV 缓存的同时，只做了极少量的 FLOPs。**所以在注意力计算中，我们基本上总是受内存带宽限制！**

<p markdown=1 class="takeaway">**要点：** 在预填充期间，对于任何合理的序列长度（大约 $\gt 480$ 个 token），注意力通常是计算受限的；而在生成期间，我们的算术强度低且恒定，因此总是受内存带宽限制。</p>

*从概念上讲，这是为什么？* 主要原因在于，模型的线性部分之所以是计算受限的，是因为参数（占用大量内存带宽的部分）被许多批次元素复用。然而，每个批次元素都有自己的 KV 缓存，因此批大小越大就意味着 KV 缓存越多。除非对架构做激进的调整，否则我们在这里几乎*总是*内存受限的。

这也意味着，一旦参数内存与 KV 缓存内存相当，再增大批大小带来的吞吐量收益就会递减。收益递减对你的影响程度取决于单个序列的参数字节数与 KV 缓存字节数之比，大致就是 $2DF / SHK$ 这个比值。由于 $HK\approx D$，它大致取决于 $F$ 与序列长度 $S$ 之比。它也取决于那些让 KV 缓存变小的架构改动（我们稍后会详细介绍）。

### LLM 延迟与吞吐量的理论估计 {#theoretical-estimates-for-llm-latency-and-throughput}

根据上面的推导，我们可以为优化时应追求的步骤时间得出相当不错的界限。**（注意：如果说我们希望读者从整章中只记住一件事，那就是下面这些内容。）** 对于生成期间的小批大小（这很常见），我们可以假设注意力块和 MLP 块都受内存带宽限制，从而给出每步延迟的下界：

$$\begin{equation*}
\text{理论最小步时间} = \frac{\text{批大小} \times \text{KV 缓存大小} + \text{参数大小}}{\text{总内存带宽}}
\end{equation*}$$

类似地，对于吞吐量：

$$\begin{equation*}
\text{理论最大 Tokens/s} = \frac{\text{批大小} \times \text{总内存带宽}}{\text{批大小} \times \text{KV 缓存大小} + \text{参数大小}}
\end{equation*}$$

最终，随着批大小的增长，FLOPs 开始超过参数加载成为主导，因此在实践中我们有一个更一般的公式：

$$\begin{align}
\tiny \text{理论步时间（一般情况）} = \underbrace{\frac{\text{批大小} \times \text{KV 缓存大小}}{\tiny \text{总内存带宽}}}_{\text{注意力（始终带宽受限）}} + \underbrace{\max\left(\frac{2 \times \text{批大小} \times \text{参数量}}{\text{总 FLOPs/s}}, \frac{\text{参数大小}}{\text{总内存带宽}}\right)}_{\tiny \text{MLP（可能计算受限）}}
\end{align}$$

其中注意力部分（左侧）永远不会是计算受限的，因此不需要 FLOPs 的 Roofline。这些公式对于粗略估算非常有用，例如：

<b markdown=1 style="color: #57cf57;">小测验：</b>假设我们想在 TPU v5e 4x4 切片上，对一个 30B 参数的稠密模型执行一次批大小为 4 个 token 的生成步骤，参数为 int8、FLOPs 为 bf16，上下文长度 8192，KV 缓存为每 token 100 kB。这个操作延迟的合理下界是多少？如果我们想采样一个 256 个 token 的批次呢？

{% details 点击此处查看答案。 %}

**答案：** 使用 int8 时，参数占用 30e9 字节；按照给定的规格，每个 KV 缓存占用 `100e3 * 8192 = 819MB`。我们有 16 个芯片，每个芯片的带宽为 `8.2e11` 字节/秒，bf16 算力为 `1.97e14` FLOPs/s。根据上面的公式，由于批大小很小，我们预计步骤时间至少为 `(4 * 819e6 + 30e9) / (16 * 8.2e11) = 2.5 ms`。在 256 个 token 时，MLP 块将完全进入计算受限区间，因此步骤时间大约为 `(256 * 819e6) / (16 * 8.2e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`。

{% enddetails %}

如你所见，这里吞吐量和延迟之间存在明显的权衡。小批次速度快，但无法充分利用硬件；大批次速度慢，但效率高。下面是为一些较早的 PaLM 模型计算出的延迟-吞吐量 Pareto 前沿（来自 [ESTI 论文](https://arxiv.org/pdf/2211.05102)<d-cite key="esti"></d-cite>）：

{% include figure.liquid path="assets/img/latency-cost.png" class="img-fluid" caption="<b>图：</b>多个 PaLM 模型的成本（即吞吐量）与延迟的 Pareto 前沿。注意芯片数量（C）和批大小（B）是如何让你沿着 Pareto 前沿移动的，唯一的例外是绿点（PaLM 540B 的 C:32 B:16），在该配置下可用内存不足以支撑合适的批大小，导致吞吐量受损。注意吞吐量通常在批大小超过 240 左右后趋于平缓。int8 权重能提供更好的延迟-吞吐量 Pareto 最优，但不能提供更高的最大吞吐量。" %}

我们不仅可以把批大小当作旋钮来权衡延迟和吞吐量；如果发现自己受限于 HBM，我们还可能更倾向于使用更大的拓扑而非更小的拓扑，以便容纳更大的批次。[下一节](../applied-inference)会对此进行更详细的探讨。

<p markdown=1 class="takeaway">**要点：** 如果你关心生成吞吐量，就使用尽可能大的每芯片批大小。任何超过 TPU 算术强度（$B_\text{crit}$，通常为 120 或 240）的每芯片批大小都能使吞吐量最大化。为此你可能需要扩大拓扑。较小的批大小能让你以牺牲吞吐量为代价改善延迟。</p>

{% details 从硬件的角度看，这里有一些需要注意的地方。点击此处查看一些细节。 %}

这些都相当理论化。在实践中，由于以下几个原因，我们通常看不到一个非常清晰的 Roofline：

* 我们假设 HBM 读取能与 FLOPs 完美重叠，这并不现实，因为我们的编译器（XLA）并非万无一失。
* 对于已分片的模型，XLA 也常常无法把模型分片矩阵乘法的 ICI 通信与 FLOPs 本身高效重叠，因此当 $$\text{BS}=32$$ 以上时，线性层往往开始出现延迟损失。
* 由于重叠不完美，大于理论 Roofline 拐点的批大小仍会看到一些吞吐量的提升，但这仍是一个不错的经验法则。

{% enddetails %}

### 内存呢？ {#what-about-memory}

我们已经花了不少篇幅讨论带宽和 FLOPs，却还没谈到内存。由于引入了新的数据结构——KV 缓存，推理时的内存状况与训练时大不相同。在本节中，我们选一个真实模型（LLaMA 2-13B）来展示两者的差别有多大：

| 超参数             | 值     |
| ------------------ | ------ |
| L (num_layers)     | 40     |
| D (d_model)        | 5,120  |
| F (ffw_dimension)  | 13,824 |
| N (num_heads)      | 40     |
| K (num_kv_heads)   | 40     |
| H (qkv_dim)        | 128    |
| V (num_embeddings) | 32,000 |

推理期间是什么在占用内存？显然，首先是我们的参数。把它们数一遍，可得：

| 参数             | 公式                                                                                                             | 大小（字节）                                                   |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| FFW 参数         | d_model<sup>2</sup> x ffw_multiplier x 3（SwiGLU 的 gate、up 和 down 投影）x n_layers                            | 5,120 x 5,120 x 2.7 x 3 x 40 = **8.5e9**                       |
| 词表参数         | 2（输入和输出嵌入）x n_embeddings x d_model                                                                      | 2 x 32,000 x 5,120 = **0.3e9**                                 |
| 注意力参数       | [2（*q 和输出*）x d_model x n_heads x d_qkv + 2（*k 和 v*）x d_model x n\_kv\_heads x d_qkv] x n_layers          | (2 x 5,120 x 40 x 128 + 2 x 5,120 x 40 x 128) x 40 = **4.2e9** |

把这些参数加起来，得到 8.5e9 + 4.2e9 + 0.3e9 = **总计 13e9 个参数**，与预期一致。正如前几节所见，训练时我们可能以 bfloat16 存储参数，并以 float32 存储优化器状态，这大约要用 100GB 内存。但与梯度检查点相比，这点内存简直不值一提——后者可能占用数 TB。

**推理有什么不同？** 推理期间，我们只存储一份参数，假设用 bfloat16，占 26GB——而实际上借助量化，我们常常能做得比这好得多。没有需要追踪的优化器状态或梯度。由于我们不做检查点（即不为反向传播保留激活值），无论是预填充<d-footnote>这尤其要归功于 Flash Attention，它避免了实体化注意力矩阵</d-footnote>还是生成，激活值占用都可以忽略不计。如果我们预填充 8k 个 token，单个激活值只占用大约 `8,192 x 5,120 x 2 bytes = 80MB` 的内存。更长的预填充可以拆分成许多较小的前向传播，所以对于更长的上下文也不是问题。生成阶段用到的 token 比这还少，因此激活值可以忽略。

**主要区别在于 KV 缓存**。它保存了所有过去 token 的 key 和 value 投影，其大小只受最大允许序列长度的限制。$$T$$ 个 token 的总大小为

$$\text{KV 缓存大小} = 2 \cdot \text{每个浮点数的字节数} \cdot H \cdot K \cdot L \cdot T$$

其中 $$H$$ 是每个头的维度，$$K$$ 是 KV 头的数量，$$L$$ 是层数，因子 2 来自同时存储 key 和 value。

**即使批大小和上下文长度都不算大，它也会迅速膨胀**。对于 LLaMA-13B，单条长度为 8192 的序列在 bf16 下的 KV 缓存为

$$8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{字节}) \times 2 = 6.7 \text{GB}$$

**只需 4 条这样的序列，就超过了参数的内存占用！** 需要说明的是，LLaMA 2 并没有针对长上下文下的 KV 缓存大小做优化（情况并不总是这么糟，因为通常 $K$ 要小得多，比如 LLaMA-3），但这个例子仍然很有说明性。在估算内存或延迟时，我们不能忽略 KV 缓存。

### 为 LLaMA 2-13B 建模吞吐量和延迟 {#modeling-throughput-and-latency-for-llama-2-13b}

我们来看看，如果在 8 块 TPU v5e 上以不同批大小、以理想的效率执行生成，一直到前面推导出的、可达到理论最大吞吐量的临界批大小（240），会发生什么。

| 批大小                            |      1 |      8 |     16 |     32 |     64 |    240 |
| :-------------------------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| KV 缓存内存（GiB）                |    6.7 |   53.6 |  107.2 |  214.4 |  428.8 |   1608 |
| 总内存（GiB）                     |   32.7 |   79.6 |  133.2 |  240.4 |  454.8 |   1634 |
| 理论步时间（ms）                  |   4.98 |  12.13 |  20.30 |  36.65 |  69.33 | 249.09 |
| 理论吞吐量（tokens/s）            | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |

8 块 TPU v5e 为我们提供 128GiB 的 HBM、6.5TiB/s 的 HBM 带宽（每块 0.82TiB/s）以及 1600TF/s 的算力。

对于这个模型，增大批大小确实能提高吞吐量，但收益递减得很快。批大小超过 16 就会 OOM，而要接近 240 则需要多一个数量级的内存。更大的拓扑可以改善延迟，但单芯片吞吐量已经碰壁了。

假设我们保持参数总数不变，但神奇地把 KV 缓存缩小到原来的 1/5（比如采用 1:5 的 [GMQA](#tricks-for-improving-generation-throughput-and-latency)，即 8 个 KV 头由 40 个 Q 头共享——详见下一节）。

| 批大小                            |      1 |        8 |       16 |       32 |       64 |      240 |
| :-------------------------------- | -----: | -------: | -------: | -------: | -------: | -------: |
| KV 缓存内存（GiB）                |   1.34 |    10.72 |    21.44 |    42.88 |    85.76 |    321.6 |
| 总内存（GiB）                     |  27.34 |    36.72 |    47.44 |    68.88 |   111.76 |    347.6 |
| 理论步时间（ms）                  |   4.17 |     5.60 |     7.23 |    10.50 |    17.04 |    52.99 |
| 理论吞吐量（tokens/s）            | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |

KV 缓存变小后，收益依然递减，但单芯片的理论吞吐量能一直提升到批大小 240。我们可以容纳大得多的批大小 64，而且在所有批大小下延迟也都更好。延迟、最大吞吐量和最大批大小全都大幅改善！事实上，后续几代 LLaMA 正是采用了这一优化——LLaMA-3 8B 有 32 个 query 头和 8 个 KV 头（[来源](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)）。

<p markdown=1 class="takeaway">**要点：** 除了参数之外，KV 缓存的大小对模型最终的推理性能也有很大影响。我们希望结合架构决策和运行时优化，把它控制在合理范围内。</p>

## 提升生成吞吐量和延迟的技巧 {#tricks-for-improving-generation-throughput-and-latency}

自最初的 [Attention is All You Need 论文](https://arxiv.org/abs/1706.03762)以来，人们已经开发出许多让模型更高效的技术，其中很多专门针对 KV 缓存。一般来说，更小的 KV 缓存使我们更容易在不损害延迟的前提下增大生成步骤的批大小和上下文长度，也让 Transformer 周边的系统（比如请求缓存）更好做。暂不考虑对质量的影响，我们可以看到以下技术：

**分组多查询注意力（Grouped multi-query attention，又称 GMQA、GQA）：** 我们可以减少 KV 头的数量，让注意力机制中的多个 Q 头共享它们。在极端情况下，可以让所有 Q 头共享单个 KV 头。与纯 MHA 相比，这能把 KV 缓存缩小为原来的 1/(Q:KV 比值)，而且人们观察到，模型性能对这一改动相对不敏感。

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

这也实际上提高了注意力计算的算术强度（参见[第 4 部分](../transformers)中的问题 4）。

**混入一些局部注意力层：** 局部注意力把上下文限制在一个小到中等的最大长度内。在训练和预填充时，这意味着把注意力矩阵掩码成一条对角带，而不是一个三角形。这实际上限定了局部层 KV 缓存的最大长度。通过在模型中把一些局部层与一些全局层混合使用，当上下文长于局部窗口时，KV 缓存的大小会大幅减小。

**跨层共享 KV：** 模型可以学会按某种模式在层间共享同一份 KV 缓存。虽然这确实能减小 KV 缓存大小，并在增大批大小、缓存、离线存储等方面带来好处，但共享的 KV 缓存可能需要从 HBM 中读取多次，*因此它不一定能改善步时间。*

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="
 <b>左：</b>多层纯全局注意力。<b>右：</b>一种全局/局部交错模式的示例，其中相邻层之间共享 KV。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai 博客</a>。"%}

**量化：** 推理通常对参数和 KV 的精度不太敏感。通过对参数和 KV 缓存做量化（例如量化到 int8、int4、`fp8` 等），我们可以节省两者的内存带宽，降低达到计算 Roofline 所需的批大小，并节省内存以便在更大的批大小下运行。量化还有一个额外的优势：即使模型训练时没有使用量化，通常也可以在训练后再施加量化。

**使用不规则（ragged）HBM 读取和 Paged Attention：** 在上面的计算中，我们为每个 KV 缓存分配了 8k 的上下文，但通常并不需要从内存中读取整个 KV 缓存——请求的长度分布范围很广，也不会用满模型的最大上下文，因此我们常常可以实现只读取 KV 缓存中非填充部分的 kernel（例如 Flash Attention 的各种变体）。

Paged Attention<d-cite key="paged"></d-cite> 是在此基础上的改进：它把 KV 缓存存储在操作系统风格的页表中，基本上完全避免了对 KV 缓存的填充。这会带来很大的复杂性，但意味着每个批次只使用它实际需要的内存。这是一种运行时优化，因此同样与架构无关。

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid img-small" caption="<b>图：</b>生成期间，单个 token（“forth”）会关注多个 KV 缓存块/页。通过对 KV 缓存分页，我们避免了加载或存储超出所需的内存。取自 <a href=\"https://arxiv.org/pdf/2309.06180\">PagedAttention 论文</a>。" %}

<p markdown=1 class="takeaway">**全局视角：** 总的来说，与标准的 MHA Transformer 相比，这些 KV 缓存优化可以把 KV 缓存大小缩小一个数量级以上。这可以让 Transformer 的总体成本改善一个数量级。</p>

## 在多个加速器上分布推理 {#distributing-inference-over-multiple-accelerators}

到目前为止，我们对如何扩展到单芯片之外一直含糊其辞。参照[第 5 部分](../training)，我们来探讨可用的各种策略及其权衡。和往常一样，我们会分别讨论预填充和生成。

### 预填充 {#prefill}

从 Roofline 的角度看，**预填充与训练几乎完全相同**，几乎所有相同的技术和权衡都适用——模型（Megatron）并行、序列分片（对于足够长的上下文）、流水线，甚至 FSDP 都是可行的！你只需把 KV 保留下来，以便之后做生成。与训练一样，增加芯片数量能让我们获得更多 FLOPs/s（可能降低 TTFT），但会增加通信开销（可能降低单芯片吞吐量）。

**预填充分片的一般规则：** 下面是一组适用于预填充的通用规则。我们假设只对单条序列做预填充（没有批维度）：

1. *模型分片：* 我们通常先做一定程度的模型并行，直到变成 ICI 受限为止。正如我们在[第 5 部分](../training)中所见，对于 1 个轴，这个界限大约是 $F / 2200$（通常约为 4-8 路分片）。
2. *序列并行：* 在此之外，我们做序列并行（类似数据并行，但沿序列维度分片）。虽然序列并行会在注意力中引入一些额外通信，但在较长的上下文下这部分通常相当小。与训练一样，我们可以让通信与计算重叠（Megatron 使用集合矩阵乘法，注意力则使用 ring attention）。

<p markdown=1 class="takeaway">**要点：** 在预填充期间，几乎任何在训练中可行的分片方式都能很好地工作。先做模型并行直到 ICI 界限，然后做序列并行。</p>

### 生成 {#generation}

生成比预填充复杂得多。一方面，获得大批大小更难，因为我们需要把许多请求批处理在一起。延迟目标也更低。这两点加在一起，意味着我们通常更偏向内存受限，并且对通信开销更敏感，这限制了我们的分片策略：

1. **FSDP 不可行：** 由于我们在把参数和 KV 缓存从 HBM 加载到 MXU 时是内存受限的，我们不希望通过比 HBM 慢几个数量级的 ICI 来移动它们。*我们要移动的是激活值，而不是权重。* 这意味着类似 FSDP 的方法在生成中通常完全不可行。<d-footnote>训练结束后不小心把它留着没关，是一种很容易发生也很常见的、导致性能倒退一个数量级的方式</d-footnote>

2. **没有理由做数据并行：** 纯数据并行没有帮助，因为它会复制参数，并不能帮我们更快地加载参数。你不如直接启动多个模型副本。<d-footnote>我们的意思是，以较小的批大小启动多个持有模型副本的服务器。在模型层面做数据并行严格劣于这种做法。</d-footnote>

3. **没有序列 = 没有序列分片。** 想做序列分片？祝你好运。

*对于稠密模型的生成，这基本上只剩下各种模型分片的变体*。与预填充一样，我们能做的最简单的事情就是简单的模型并行（激活值完全复制，MLP 的权重沿隐藏维度完全分片），做到 4-8 路，此时会变成 ICI 受限。然而，由于我们常常是内存带宽受限的，实际上可以超出这个界限来改善延迟！

**关于生成中 ICI 界限的说明：** 训练时我们希望是计算受限的，所以 Roofline 分析关注的是 ICI 通信何时比 FLOPs 耗时更长。然而在生成期间，如果我们受限于参数加载的内存带宽，就可以把模型分片增加到超过这一点，以极小的吞吐量代价（以 tokens/sec/chip 计）改善延迟。更多的模型分片意味着有更多的 HBM 可用来加载权重，而 FLOPs 并不重要。<d-footnote>意思是 FLOPs 时间并不是瓶颈，所以我们需要担心的是 ICI 时间超过参数加载时间。</d-footnote> 我们来看看在 ICI 通信成为瓶颈之前，我们能做多少路模型并行。

$$\begin{align*}T_\text{HBM 通信} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI 通信} = \frac{2BD}{W_\text{ici}}\end{align*}$$

$$T_\text{ICI 通信} > T_\text{HBM 通信} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)$$

其中 $\beta = W_\text{hbm} / W_\text{ici}$。对于 TPU v5e 和 TPU v6e，这个数通常约为 8。这意味着，例如当 $F$ 为 16,384、$B$ 为 32 时，理论上我们可以做到 `16384 / (32 * 8) = 64` 路模型并行，而不会对吞吐量造成明显影响。这里假设我们能把 KV 缓存完全做 64 路分片，而这很难做到：我们将在下文讨论。

对于注意力层，我们同样以 Megatron 的方式沿头维度对注意力的 $$W_Q$$ 和 $$W_O$$ 做模型分片。KV 权重相当小，与超出 $K$ 路的分片相比，复制它们往往更划算。

<p markdown=1 class="takeaway">**要点：** 在生成期间，我们唯一的选择是各种模型并行的变体。我们的目标是移动激活值，而不是更大的 KV 缓存或参数。当批大小较大时，我们把模型并行做到 FLOPs-ICI 界限（$F / \alpha$）。当批大小较小时，我们可以通过更多的模型分片来改善延迟（代价是适度的吞吐量损失）。当我们想要的模型分片路数超过 KV 头的数量时，还可以沿批维度对 KV 进行分片。</p>

### 对 KV 缓存分片 {#sharding-the-kv-cache}

**我们还有一个需要分片的额外数据结构——KV 缓存。** 同样，我们几乎总是倾向于避免复制缓存，因为它是注意力延迟的主要来源。为此，我们首先沿头维度对 KV 做 Megatron 分片。这最多只能做 $K$ 路分片，因此对于头数较少的模型，我们会尽可能多地对头维度分片，然后再沿批维度分片，即 $\text{KV}[2, B_Z, S, K_Y, H]$。这意味着 KV 缓存是完全分布式的。

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>图：</b>注意力机制的对比：(a) 采用纯模型分片的多头注意力；(b) 对 KV 缓存做批分片的多查询注意力。注意我们需要两次额外的 AllToAll，把激活值从模型分片切换为批分片，这样它们才能作用于 KV 缓存。" %}

其代价是每个注意力层需要两次 AllToAll——一次把 Q 激活值切换为批分片，以便在批分片下计算注意力；另一次把批分片的注意力输出切换回纯模型分片。

{% details 完整算法见此！ %}

这里我们写出在 $Y$ 和 $Z$ 两个轴上都做模型并行时的完整注意力算法。抱歉，我同时用 $K$ 表示 key 张量和 KV 头维度。令 $M=N/K$。

<div markdown=1 class="algorithm">

1. X[B, D] = ...（已有的激活值，来自上一层，未分片）
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K<sub>Y</sub>, H] = ...（已有的 KV 缓存，按批分片）
3. Q[B, N<sub>YZ</sub>, H] = X[B, D] \* W<sub>Q</sub>[D, N<sub>YZ</sub>, H]
4. Q[B<sub>Z</sub>, N<sub>Y</sub>, H] = **AllToAll**<sub>Z->B</sub>(Q[B, N<sub>YZ</sub>, H])
5. Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = **Reshape**(Q[B<sub>Z</sub>, N<sub>Y</sub>, H])
6. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] \*<sub>H</sub> K[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
7. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = **Softmax**<sub>S</sub>(O[B<sub>Z</sub>, S, K<sub>Y</sub>, M])
8. O[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] \*<sub>S</sub> V[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
9. O[B, K<sub>Y</sub>, M<sub>Z</sub>, H] = **AllToAll**<sub>Z->M</sub>(O[B<sub>Z</sub>, K<sub>Y</sub>, M, H])
10. O[B, N<sub>YZ</sub>, H] = **Reshape**(O[B, K<sub>Y</sub>, M<sub>Z</sub>, H])
11. X[B, D] {U<sub>YZ</sub>} = W<sub>O</sub>[N<sub>YZ</sub>, H, D] \*<sub>N,H</sub> O[B, N<sub>YZ</sub>, H]
12. X[B, D] = **AllReduce**(X[B, D] { U<sub>YZ</sub>})

这相当复杂，但你大致能看出它是如何工作的。新增的通信开销不算太大，因为它们作用于我们较小的激活值；作为回报，我们在加载 KV（它们是静止不动的）时节省了大量内存带宽。

</div>

{% enddetails %}

* **序列分片：** 如果批大小太小，或者上下文很长，我们可以对 KV 缓存做序列分片。同样，这里我们要为跨分片累积注意力付出集合通信的代价。首先需要对 Q 激活值做 AllGather，然后以类似 Flash Attention 的方式累积 KV。

## 设计高效的推理引擎 {#designing-an-effective-inference-engine}

到目前为止，我们研究的是如何孤立地对单独的预填充和生成操作进行高效的优化和分片。要真正有效地使用它们，我们需要设计一个推理引擎，让它能在延迟/吞吐量的帕累托前沿上我们所选的某个点上驱动这两种操作。

最简单的方法就是先运行一批预填充，再运行一批生成：

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" caption="<b>图：</b>在最简单的设置中，请求被聚合起来，服务器交替执行：运行一批预填充，然后调用生成函数，直到所有序列都完成。" %}

这很容易实现，也是大多数代码库中的第一个推理设置，但它有多个缺点：

1. **延迟很糟糕。** 我们把预填充和生成的批大小耦合在了一起。在预填充批大小较大时，首 token 时间（TTFT）很糟糕——你需要完成所有预填充后，任何用户才能看到任何 token。而在批大小较小时，生成吞吐量又很糟糕。
2. **较短的生成会被较长的生成阻塞。** 许多序列会比其他序列先结束，在生成期间留下空的批槽位，进一步损害生成吞吐量。批大小和生成长度越大，这个问题越严重。
3. **预填充会被填充（padding）。** 预填充要填充到最长序列的长度，浪费大量计算。这个问题有解决办法，但历史上 XLA 很难跳过这些 FLOPs。同样，批大小和预填充序列长度越大，情况越糟。
4. **我们被迫在预填充和生成之间共用一种分片方式。** 预填充和生成都位于同一个切片上，这意味着两者使用相同的拓扑和分片方式（除非你保留两份权重），这通常不利于性能，例如生成需要多得多的模型分片。

因此，只推荐在边缘应用（通常只需服务单个用户，且使用的硬件 FLOPs/byte 较低）以及 Transformer 代码库生命周期早期的快速迭代（因为它简单）中使用这种方法。

一种稍好的方法是以批大小 1 执行预填充（此时它是计算受限的，延迟也合理），但在生成期间把多个请求批处理在一起：

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

这避免了批量预填充造成的 TTFT 浪费，同时保持较高的生成吞吐量。我们称之为**交错（interleaved）**配置，因为我们把预填充和生成步骤“交错”进行。这对于评测这类以吞吐量为主要目标的批量生成应用非常有效。编排器可以配置为：一旦有生成槽位空出来，就优先执行预填充，从而即使在非常大的生成批大小下也能确保高利用率。我们还可以避免把预填充填充到最大长度，因为它不会与其他请求批处理在一起。

主要缺点是，当服务器执行预填充时，所有其他请求的生成都会暂停，因为所有计算资源都被预填充占用了。正在解码响应的用户 A 会被正在进行预填充的用户 B 阻塞。这意味着虽然 TTFT 改善了，但 token 生成会出现抖动，平均速度也较慢，这对许多应用来说都不是好的用户体验——其他用户的预填充处在请求总体延迟的关键路径上。

为了解决这个问题，我们把解码和预填充分开。虽然 Transformer 推理可以在一台服务器上完成，但从延迟的角度看，在两组 TPU/GPU 上分别执行这两种不同任务往往更好。预填充服务器生成 KV 缓存，通过网络发送到生成服务器，后者把多个缓存批处理在一起，并为每个缓存生成 token。我们称之为**“分离式”**部署（disaggregated serving）。

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

这带来了几个优势：

1. **大规模下的低延迟**：一个用户的请求永远不会被另一个用户的请求阻塞，除非预填充容量不足。请求应当立即被预填充，然后发送到生成服务器，再立即插入生成缓冲区。如果我们预计会有大量并发请求涌入，可以独立于生成服务器的数量来扩展预填充服务器的数量，这样用户就不会在预填充队列中滞留太久。

2. **专门化：** 很多时候，预填充和生成在延迟上最优的参数分片策略/硬件拓扑差别很大（例如，更多的模型并行对生成有用，对预填充则不然）。强制两种操作使用同一种分片方式会损害两者的性能，而保留两套权重又会占用内存。此外，把预填充移到单独的服务器上后，它除了当前正在处理的 KV 缓存之外，不需要保存任何 KV 缓存。这意味着我们有多得多的空闲内存可用于历史缓存（见下一节）或优化预填充延迟。

一个缺点是 KV 缓存现在需要通过网络传输。这通常是可以接受的，但这也再次说明了减小 KV 缓存大小的意义。

<p markdown=1 class="takeaway">**要点：** 对于延迟敏感、高吞吐量的部署，我们通常必须把预填充和生成分到不同的服务器上，预填充以批大小 1 运行，生成则把许多并发请求批处理在一起。</p>

### 连续批处理 {#continuous-batching}

上面的问题 (2) 引出了**连续批处理（continuous batching）**的概念。我们优化并编译：

* 一个预填充函数，它能处理可变的上下文长度，并把结果插入一个具有某个最大批大小和上下文长度/页数的 KV 缓冲区。
* 一个生成函数，它接收 KV 缓存，并为所有当前活跃的请求执行生成步骤。

然后，我们把这些函数与一个编排器结合起来：编排器对传入的请求进行排队，根据可用的生成槽位调用预填充和生成，处理历史缓存（见下一节），并把 token 流式输出。

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

### 前缀缓存 {#prefix-caching}

由于预填充开销大且是计算受限的（留给我们的余量更小），降低其成本的最佳方法之一就是少做预填充。因为 LLM 是自回归的，查询 ["I", "like", "dogs"] 和 ["I", "like", "cats"] 产生的 KV 缓存在前两个 token 上是完全相同的。这意味着，原则上如果我们先计算“I like dogs”的缓存，再计算“I like cats”的缓存，只需要做 1 / 3 的计算量。通过复用缓存，我们可以省下大部分工作。这在以下几种特定场景中尤其有效：

1. **聊天机器人**：大多数聊天机器人对话都是一来一回的，并且严格地在已有内容之后追加。这意味着如果我们能保存每一轮对话的 KV 缓存，就可以跳过除最新 token 之外所有 token 的计算。
2. **少样本提示（few-shot prompting）**：如果我们有任何形式的少样本提示，都可以把它保存下来并免费复用。系统指令通常也具有这种形式。

唯一让这件事变难的原因是内存限制。正如我们所见，KV 缓存很大（通常有好几 GB），而要让缓存发挥作用，我们需要一直保留它们，直到后续查询到来。通常，预填充服务器上任何未使用的 HBM 都可以用于本地缓存系统。此外，加速器的 CPU 主机上通常也有大量内存（例如一台 8xTPUv5e 服务器有 128GiB 的 HBM，但有大约 450GiB 的主机 DRAM）。这部分内存比 HBM 慢得多——通常慢到无法用来执行生成步骤——但对于缓存读取来说已经足够快。在实践中：

* 由于 KV 缓存位于处理初始请求的那组 TPU 本地，我们需要某种形式的亲和性路由（affinity routing）来确保后续查询到达同一个副本。这可能会给负载均衡带来问题。
* 更小的 KV 缓存（再一次）很有帮助——它让我们能在同样大小的空间里保存更多 KV 缓存，并减少读取时间。
* KV 缓存及其查找可以很自然地存储在树或字典树（trie）中。淘汰可以基于 LRU 策略进行。

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>图：</b>以 LRU 字典树实现的 KV 前缀缓存。通过共享前缀，我们可以避免重复存储 KV 内存。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai 博客</a>。" %}

### 来看一个实现：JetStream {#lets-look-at-an-implementation-jetstream}

Google 开源了一个实现上述逻辑的库，名为 [JetStream](https://github.com/google/JetStream)。服务器有一组“预填充引擎”和“生成引擎”，通常位于不同的 TPU 切片上，由单个控制器统一编排。预填充在“[预填充线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)”中进行，而生成在“[生成线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)”中进行。我们还有一个“[传输线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)”，负责编排把 KV 缓存从预填充切片复制到生成切片。

Engine 接口（实现见[这里](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)）是任何 LLM 都必须提供的通用接口。关键方法有：

* **prefill：** 接收一组输入 token 并生成 KV 缓存。
* **insert：** 接收一个 KV 缓存，并把它插入到 generate 正在用于生成的那批 KV 缓存中。
* **generate：** 接收一组批处理的 KV 缓存，为批中的每个条目生成一个 token，并为每个 token 把单个 token 的 KV 缓存追加到解码状态中。

我们还提供了 JetStream 的 PyTorch 版本，见[这里](https://github.com/google/jetstream-pytorch)。

## 练习题 {#worked-problems}

在这一节中，我将基于 LLaMA-2 13B 虚构一个新模型。细节如下：

| 超参数             | 取值   |
| :----------------- | :----- |
| L (num_layers)     | 64     |
| D (d_model)        | 4,096  |
| F (ffw_dimension)  | 16,384 |
| N (num_heads)      | 32     |
| K (num_kv_heads)   | 8      |
| H (qkv_dim)        | 256    |
| V (num_embeddings) | 32,128 |

**问题 1：** 上述模型有多少参数？在 int8 下，每个 token 的 KV 缓存有多大？*你可以假设输入和输出投影矩阵是共享的。*

{% details 点击此处查看答案。 %}

**参数量：**

* MLP 参数量：$L * D * F * 3$
* 注意力参数量：$L * 2 * D * H * (N + K)$
* 词表参数：$D * V$（因为我们共享这些矩阵）

因此总参数量为 $L * D * (3F + 2H * (N + K)) + D * V$。代入上面的数字，得到 `64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`。因此，这个模型大约有 184 亿个参数。

在 int8 下，KV 缓存为每 token $2 * L * K * H$，即每 token `2 * 64 * 8 * 256 = 262kB`。

{% enddetails %}

**问题 2：** 假设我们想在 TPUv5e 4x4 切片上部署这个模型，并且可以把 KV 缓存在这个拓扑上完全分片。假设所有东西都用 int8，并且要支持 128k 长度的序列，我们能容纳的最大批大小是多少？如果把 KV 头数降到 1 呢？

{% details 点击此处查看答案。 %}

在 int8 下，KV 缓存大小为每 token $2 \cdot L \cdot K \cdot H$，即 `2 * 64 * 8 * 256 = 262kB`。对于 128k 长度的序列，这意味着每个批条目需要 `262e3 * 128e3 = 33.5GB`。由于每个 TPU 有 16GB 的 HBM，算上参数，我们能容纳的最大批大小为 `(16 * 16e9 - 18.4e9) / 33.5e9 = 7`。如果 $K=1$，则是这个值的 8 倍，也就是大约 56。

{% enddetails %}

**问题 3：** 假设参数在 TPU v5e 4x4 切片上完全分片，把所有参数从 HBM 加载到 MXU 需要多长时间？假设参数为 int8。*这是每步延迟的一个很好的下界。*

{% details 点击此处查看答案。 %}

我们总共有 18.4B 个参数，在 int8 下即 18.4e9 字节。每个芯片的 HBM 带宽为 8.2e11，因此假设我们能完全利用 HBM 带宽，大约需要 `18e9 / (8.2e11 * 16) = 1.4ms`。

{% enddetails %}

**问题 4：** 假设我们想在 TPUv5e 4x4 切片上用 int8 FLOPs 和 int8 参数/激活值部署这个模型。在预填充和解码阶段分别应该如何分片？*提示：也许可以先回答下面这些问题：*

1. 4x4 上的 ICI 是什么样的？
2. 张量并行的 Roofline 界限是多少？
3. 我们可以如何对 KV 缓存进行分片？

对于这种分片方式，生成阶段每步的大致延迟是多少？

**问题 5：** 假设上面的模型实际上是一个 MoE。MoE 模型实际上就是一个拥有 E 份 FFW 块副本的稠密模型。每个 token 会经过其中 k 个 FFW 块，这 `k` 个块的输出取平均后得到最终输出。我们在上述设置的基础上取 `E=16` 和 `k=2`。

1. 它的总参数量和激活参数量各是多少？*激活参数指的是任意给定 token 会用到的参数。*
2. 在 TPU v5e 上需要多大的批大小才能达到计算受限（FLOPs bound）？
3. 每个 token 的 KV 缓存有多大？
4. 对 T 个 token 做一次前向传播涉及多少 FLOPs？

{% details 点击此处查看答案。 %}

(1) 作为 MoE，每个 MLP 块现在有 $3 * E * D * F$ 个参数，是稠密版本的 $E$ 倍。因此它现在共有 $L * D * (3EF + 2H * (N + K)) + D * V$，即 `64 * 4096 * (3*16*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 212e9` 个总参数，增加了约 12 倍。对于激活参数，被激活的是 $k$ 个而不是 $E$ 个专家，总计 `64 * 4096 * (3*2*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 31.2e9`，比稠密版本增加不到 2 倍。

(2) 因为参数量增加了 $E$ 倍而 FLOPs 只增加了 $k$ 倍，我们的 HBM Roofline 界限提高了 $E/k$ 倍。这意味着在 TPU v5e 上我们大约需要 `240 * (16 / 2) = 1920` 个 token。

(3) KV 缓存大小保持不变，因为 MoE 的特性丝毫不改变注意力机制。

(4) 这仍然是 $2 \cdot \text{激活参数量} \cdot T$。因此为 $2 * \text{31.2e9} * T$。

{% enddetails %}

**问题 6：** 对于 MoE，我们可以做“专家分片”，即把专家拆分到网格的某一个轴上。按照我们的标准记法，第一个 FFW 权重的形状为 `[E, D, F]`，我们把它分片为 [E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>]，其中 `X` 只在训练时作为 FSDP 维度使用。假设我们想在 TPU v5e 上做推理：

1. 在 Y=8、Z=16 的 TPU v5e 8x16 切片上，上述模型的 HBM 权重加载时间是多少？每个 TPU 还剩多少空闲 HBM？
2. 能容纳这个模型的最小切片是多大？

**问题 7 [2D 模型分片]：** 这里我们来推导 [ESTI 论文](https://arxiv.org/pdf/2211.05102)中所说的 2D 权重驻留（weight-stationary）分片的数学。我们在附录 B 中对此有简要描述，但建议先尝试做这道题，看看你能否自己推出这些数学。2D 权重驻留分片的基本思想是沿 $D$ 和 $F$ 两个轴同时对权重进行分片，使每个分块大致呈方形。这能降低通信负载，让我们可以扩展得稍远一些。

下面是 2D 权重驻留的算法：

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])
2.  Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>})
4.  Out[B, D<sub>X</sub>] {U<sub>YZ</sub>} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>YZ</sub>, D<sub>X</sub>]
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U<sub>YZ</sub>})
</div>

你的目标是推导出该算法的 $T_\text{math}$ 和 $T_\text{comms}$，并找出它在什么情况下会优于传统的 3D 模型分片。

{% details 点击此处查看答案！ %}

我们来推导 $T_\text{math}$ 和 $T_\text{comms}$。所有 FLOPs 都是完全分片的，因此和之前一样有 $T_\text{math} = 4BDF / (N \cdot C)$，但通信量现在变为

$$\begin{align*}
T_\text{2D 通信} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}$$

这里需要注意，AllReduce 的开销是两倍，并且我们按每个操作所跨越的轴数对通信量进行缩放。假设我们可以自由选择拓扑，并假设 $F=4D$（如 LLaMA-2 中那样），我们断言（通过一些基础的微积分）$X$、$Y$ 和 $Z$ 的最优取值为 $X = \sqrt{N / 8}$、$YZ = \sqrt{8N}$，因此总通信量为

$$T_\text{2D 通信} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}$$

首先，照搬上文的结果，普通的 1D 模型并行有 $T_\text{模型并行通信} = 4BD / (3 \cdot W_\text{ici})$，那么新方案的通信量什么时候更小？我们有

$$\begin{align*}
T_\text{模型并行通信} > T_\text{2D 通信} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\
\iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 72
\end{align*}$$

对于一般的 $F$，我们断言该条件为

$$N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2$$

这告诉我们，如果芯片数超过 72，使用这个新方案会更好。这个结果有点奇怪，因为从历史经验来看，我们通常在大约 ~20 路张量并行时就会受限于 ICI。但在这里，即使我们是通信受限的，总通信量也会随着芯片总数的增加而持续下降！这告诉我们，可以继续增加芯片数、增大批大小、进一步扩大参数规模，同时看到延迟降低。

{% enddetails %}

<h3 markdown=1 class="next-section">第 7 部分到此结束！第 8 部分将探讨如何在 TPU 上部署 LLaMA 3，请点击[这里](../applied-inference)。</h3>

## 附录 {#appendix}

### 附录 A：批大小 > 240 的规则有多可靠？ {#appendix-a-how-real-is-the-batch-size--240-rule}

我们在上文给出的简单规则——批大小必须大于 240 个 token 才能达到计算受限——大体上是对的，但它忽略了 TPU 在其他操作没有用满全部 HBM 带宽时（比如进行设备间通信时）预取权重的能力。

下面是一个小型 Transformer 的层耗时（单位为微秒）的实测图，该模型 d<sub>model</sub> 为 8192，d<sub>ff</sub> 为 32768，每层只有 2 次矩阵乘法。数据来自[这个 Colab 笔记本](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)。你会看到，在批大小达到大约 240 之前，步耗时增长得非常缓慢，之后则呈线性增长。

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid img-small" %}

下面是以 tokens / us 为单位的实际吞吐量。它相当清楚地说明了这一论点。由于这里的层大约有 600M 参数，并做了 4 路分片，我们预计最低延迟约为 365us。

{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid img-small" %}

所以至少在这个模型上，我们确实看到吞吐量一直增长到每个数据并行分片大约 BS240 为止。

### 附录 B：2D 权重驻留分片 {#appendix-b-2d-weight-stationary-sharding}

随着拓扑规模增大，如果我们能使用更高维度的网格（比如 TPU 的网格），就可以通过引入第二个分片轴，用“**2D 权重分片**”进一步改进。我们称之为“**2D 权重驻留**”，[Efficiently Scaling Transformer Inference 论文](https://arxiv.org/abs/2211.05102)对此有更详细的描述。

由于在 Megatron 中我们只对隐藏维度 $$F$$ 进行分片，一旦芯片数量变大，在 1D 分片下它可能会变得比 $$E$$（即 $$d_\text{model}$$ 维度）小得多。这意味着在较大的批大小下，在应用 MLP 的第一层之后，沿隐藏维度执行一部分集合通信可能会更划算。

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid img-small" %}

这张图展示了：

1. 1D 权重驻留分片，也就是纯 Megatron 分片：激活值在 AllGather 之后被完全复制，而权重沿隐藏维度 F 被完全分片。
2. 2D 权重驻留分片：权重同时沿隐藏维度 F 和归约维度 E 分片，激活值沿 E 维度分片。我们在第一层之前沿 (yz) 轴执行 AllGather，然后沿 (x) 轴执行 ReduceScatter。

对于注意力层，当芯片数量较少时，Megatron 风格的分片同样比较简单。然而，Megatron 是沿 $$n_\text{heads}$$ 维度进行的，这限制了可以进行的分片程度。把 2D 分片修改后用于注意力（不再对隐藏维度分片，而是对 $$n_\text{heads}$$ 维度分片），我们就能获得进一步扩展的能力。

### 附录 C：受延迟限制的通信 {#appendix-c-latency-bound-communications}

回顾一下，在[第 3 部分](../sharding)中，我们推导了在一个由 X 个芯片组成、链路全双工带宽为 WICI、延迟为 Tmin 的 1D 环上，执行一次 AllGather 得到每个 TPU 上大小为 B 的张量所需的时间。

$$T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)$$

对于较大的 B，实际耗时（wall clock）基本保持不变，因为当你向系统中添加更多芯片时，执行该操作所需的数据移动量和可用的总带宽会同步增长。

{% include figure.liquid path="assets/img/all-gather.gif" class="img-fluid" %}

由于在以延迟为优化目标的推理中移动的数据量相对较少，激活值上的集合通信往往受延迟项限制（尤其是在小批大小下）。我们可以很容易地直观理解延迟：数一数完成操作之前需要经过多少跳即可。

在 TPU 上，如果通信中与张量大小相关的部分每跳不到 1 微秒（一跳是指两个相邻设备之间的通信），我们就可能会被实际分派集合通信的固定开销所限制。在 `4.5e10` 的单向 ICI 带宽下，当 $$(\text{字节} / n_\text{shards}) / 4.5e10 < 1e-6$$ 时，ICI 通信就会变为受延迟限制。对于 8 路 Megatron 分片，也就是 `buffer_size < 360kB` 时。**这在推理中其实并不算小：** 在 `BS=16`、`D=8192` 且使用 int8 时，我们的激活值会占用 `16*8192=131kB`，因此已经是受延迟限制的了。

<p markdown=1 class="takeaway">**要点：** 当 $$\text{总字节数} < W_{ICI} \times 1e-6$$ 时，我们的通信会变为受延迟限制。例如，在沿 $$Y$$ 做模型并行时，在 int8 下当 $$Y > BD / 45,000$$ 时就会受延迟限制。</p>

这里可以和计算的 Roofline 模型做一个类比——我们在承担一些小操作的固定开销（通信的延迟、矩阵乘法的内存带宽）。

### 附录 D：投机采样 {#appendix-d-speculative-sampling}

当我们*真正*在意端到端延迟时，还可以使用一个额外的技巧，叫做投机采样（speculative sampling）<d-cite key="spec1"></d-cite><d-cite key="spec2"></d-cite>。回顾一下，我们通常是从一个大型 Transformer 中逐个生成 token：

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}

使用投机采样时，我们用一个更小、更便宜的模型来生成 token，然后用大模型来检验结果。这用*贪心解码（greedy decoding）*最容易理解：

{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

1. 我们从某个更小、更便宜的模型中贪心地采样。理想情况下，我们使用一个经过训练以匹配大模型的模型（例如通过蒸馏得到），但它也可以简单到只是使用 n-gram，或者在一个小型文本语料上做 token 匹配。
2. 生成 K 个 token 之后，我们用大模型为目前生成的所有 token 计算下一个 token 的 logits。
3. 由于我们是贪心解码，只需检查小模型生成的 token 是否在所有可能的 token 中概率最高即可。如果其中某个 token 是错的，我们就取最长的正确前缀，把第一个错误 token 替换为正确的 token，然后回到 (1)。如果所有 token 都正确，我们可以用最后一个正确的 logit 额外采样一个 token，再回到 (1)。

**为什么这能带来延迟上的收益？** 这个方案仍然要求我们为每个 token 做相当于大模型一次前向传播的 FLOPs，但由于我们可以把一批 token 放在一起处理，就能在一次前向传播中完成所有这些 FLOPs，并利用我们*并非计算受限*这一点，免费为更多 token 打分。

平均而言，每个被接受的 token 在 FLOPs 上都变得更贵了（因为有些会被拒绝，而且我们还要调用草稿模型），但我们从硬件中榨取了更多的 FLOPs，而且小模型很便宜，所以总体上我们是赚的。我们还在多个步骤之间分摊了 KV 缓存的加载，因此**对于长上下文，投机解码也能带来吞吐量上的收益。** 由于所有内容都经过了大模型的检验，我们完全不会改变采样分布（不过对于非贪心解码，具体的生成轨迹会有所不同）。

传统上，投机解码依赖于存在一个与目标模型采样分布相近的更小模型，例如用 LLaMA-2 2B 搭配 LLaMA-2 70B，而这样的模型往往并不存在。即使有这样的模型，如果接受率很低，较小的草稿模型（drafter）仍然可能太昂贵。另一种做法是把草稿模型嵌入到主模型内部，这会很有帮助，例如在基础模型靠后的某一层上添加一个专用的草稿头（drafter head）<d-cite key="eagle"></d-cite><d-cite key="medusa"></d-cite><d-cite key="DeepSeek3"></d-cite>。由于这个头与主模型共享大部分参数，它运行得更快，采样分布也与主模型更接近。

对于普通的自回归采样，token/s 与步耗时是一样的。我们仍然受制于这里算术强度一节中给出的理论最小步耗时（事实上，投机采样的步耗时通常比普通自回归采样慢不少，但由于平均每步能得到不止 1 个 token，我们可以获得好得多的 tokens/s）。

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>图：</b>这张图展示了 Chinchilla（DeepMind 的一个 70B 模型）搭配一个 4B 参数草稿模型（小模型）时的每步延迟和投机成功率。对于 XSum（一个自然语言数据集），理想的投机长度大约是向前 3-4 个 token；而 HumanEval（一个代码数据集）更可预测，更激进的投机能带来收益。" %}

**对于非贪心解码，这又是如何工作的？** 这会稍微复杂一些，但本质上可以归结为一个受 Metropolis-Hastings 启发的算法：我们从 logits 中得到 $$P_{\text{草稿模型}}(\text{选中的 token})$$ 和 $$P_{\text{目标模型}}(\text{选中的 token})$$，如果这两个概率之比小于某个阈值，就以一定概率拒绝所选的 token。

这[两篇](https://arxiv.org/abs/2211.17192)[论文](https://arxiv.org/abs/2302.01318)同时推导出了这一方法，并给出了它在实践中如何运作的很好的示例。

<p markdown=1 class="takeaway">**要点：** 投机采样是又一个强有力的杠杆，可以用吞吐量换取更好的单 token 延迟。然而，在批大小受限的场景下（例如硬件规模小或 KV 缓存很大），它就成了一个双赢的方案。</p>
