---
layout: distill
title: "如何对 TPU 程序做性能分析"
# permalink: /main/
description: "到目前为止，本系列完全是理论性的：基于硬件 Roofline 的粗略估算。这些理解能让你走得很远，但大量优化工作最终取决于实际细节：XLA 编译器是如何工作的，以及当它表现不佳时，如何使用 JAX/TensorBoard Profiler 这样的性能分析工具来弄清楚该怎么做。本章就来讨论这些内容。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 9

previous_section_url: "../applied-inference"
previous_section_name: "第 8 部分：部署 LLaMA"

next_section_url: ../jax-stuff
next_section_name: "第 10 部分：JAX"

giscus_comments: false
giscus_discussion: 11

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
  - name: "TPU 软件栈鸟瞰"
    id: a-thousand-foot-view-of-the-tpu-software-stack
  - name: "JAX Profiler：多用途的 TPU 性能分析器"
    id: the-jax-profiler-a-multi-purpose-tpu-profiler
  - subsections:
    - name: "Trace Viewer"
      id: trace-viewer
    - name: "如何阅读 XLA 操作"
      id: how-to-read-an-xla-op
    - name: "Graph Viewer"
      id: graph-viewer
    - name: "看一个（近乎）真实的性能分析示例"
      id: looking-at-a-realish-example-profile
    - name: "Memory Profile"
      id: memory-profile
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

## TPU 软件栈鸟瞰 {#a-thousand-foot-view-of-the-tpu-software-stack}

Google 提供了一系列用于 TPU 编程的 API，从高层的 JAX 代码到底层的 Pallas 或 HLO。大多数程序员只写 JAX 代码，它让你可以编写抽象的、NumPy 风格的线性代数程序，这些程序会被自动编译，从而在 TPU 上高效运行。

下面是一个简单的例子，一个将两个矩阵相乘的 JAX 程序：

```py
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

通过调用 `jax.jit`，我们让 JAX 对这个函数进行 trace，并生成一种更底层的 IR，叫做 [StableHLO](https://openxla.org/stablehlo)。它是一种与平台无关的、用于机器学习计算的 IR，随后会被 XLA 编译器进一步降级（lower）为 HLO。编译器会运行许多遍（pass）来确定融合（fusion）、布局（layout）以及其他因素，最终得到你在 JAX 性能分析结果中能看到的 HLO。这份 HLO 以一种 LLVM 风格的图视图表示了 JAX 代码中所有核心的线性代数操作（矩阵乘法、逐元素操作、卷积等）。例如，下面是上述程序对应 HLO 的精简版本<d-footnote>要获得这份 HLO，你可以运行 `jax.jit(f).lower(*args, **kwargs).compile().as_text()`。</d-footnote>：

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

我们马上就会解释 HLO 的语法，但现在只需注意到，它其实与上面的 JAX 代码对应得相当好。例如，

```c
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

就是上面那个实际的矩阵乘法，它分别沿第 0 维和第 1 维将两个 f32 矩阵相乘。

**为了把这份 HLO 转换成能在 TPU 上执行的代码，XLA 编译器首先将其降级为 LLO**（low-level optimizer，底层优化器）IR。LLO 直接对 TPU 编程：调度不同内存之间的拷贝、把数组推入脉动阵列（systolic array）等等。LLO 代码包含一些原语，用于将缓冲区推入脉动阵列、取出结果，以及调度在 TPU 不同内存之间进行通信的 DMA。一旦降级为 LLO，它就会再被编译成机器码，加载到 TPU 的 IMEM 中执行。

当程序运行得比我们期望的慢时，我们主要在 JAX 层面着手提升性能。然而，这样做往往需要我们理解 HLO 的一些语义，以及代码在 TPU 上实际是如何运行的。当更底层出了问题时，我们会动用另一个“逃生舱口”，用 [Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html) 编写自定义内核（kernel）。要查看程序的 HLO 及其运行时统计数据，我们使用 JAX 性能分析器（profiler）。

## JAX Profiler：多用途的 TPU 性能分析器 {#the-jax-profiler-a-multi-purpose-tpu-profiler}

JAX 提供了一个多用途的 TPU 性能分析器，其中包含许多有用的工具，帮助你理解程序运行时 TPU 上发生了什么。你可以使用 `jax.profiler` 模块在程序运行时对其进行 trace，记录从每个子组件的耗时、每个程序的 HLO、内存使用情况等方方面面的信息。例如，下面这段代码会把一份 trace 转储到 `/tmp/tensorboard` 中的文件里，可以在 TensorBoard 中查看（[这里](https://docs.jax.dev/en/latest/profiling.html#tensorboard-profiling)有一份分步指南）。

```py
import jax
with jax.profiler.trace("/tmp/tensorboard"):
  key = jax.random.key(0)
  x = jax.random.normal(key, (1024, 1024))
  y = x @ x
  y.block_until_ready()

# Now you can load TensorBoard in a Google Colab with
#
# !pip install -U xprof
# !pip install -U protobuf
# %load_ext tensorboard
# %tensorboard --logdir=/tmp/tensorboard
#
# or externally with
#
# > tensorboard --logdir=/tmp/tensorboard
#
```

下面概览了你可以在性能分析器中做些什么：

{% include figure.liquid path="assets/img/xprof-overview.png" class="img-fluid" %}

进入 TensorBoard 后，性能分析器有几个关键的标签页，可以帮助你理解程序：

1. **Trace Viewer** 展示了 TPU 上实际发生的事情的详细时间线。
2. **Graph Viewer** 展示了 HLO 图，让你看到程序各部分之间的数据流向（哪部分的输出作为哪部分的输入），以及各部分是如何分片的。
3. **Memory Profile 和 Memory Viewer：** 它们展示了你的程序使用了多少内存。

虽然分享性能分析结果有点困难，但[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)有一个 Perfetto 链接，至少包含了一个简单 Transformer 的 Trace Viewer 部分。[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 可以让你生成完整的 JAX/TensorBoard trace 并动手探索。

### Trace Viewer {#trace-viewer}

**Trace Viewer 大概是性能分析器中最有用的部分。** 下面的例子展示了一个简单的 Transformer，并标注了各个部分。名称来自代码中提供的标签。

{% include figure.liquid path="assets/img/trace-viewer.png" class="img-fluid" %}

Trace Viewer 按时间顺序展示了每个 TPU 核心上所有操作的时间线。这里我们只看 TPU:0，因为通常所有 TPU 执行的都是相同的指令。几个要点：

1. 最上面一行（XLA Ops）显示的是实际的 TPU 操作（名称是 HLO 名称）。其余所有内容都是基于 `jax.named_scope`、`jax.named_call` 和 Python 调用栈的近似 trace。
2. 注意到重复出现的块，我们可以在这里单独分离出一层。我们还能看出（通过查看代码 / 理解 Transformer 的工作原理）哪些部分是注意力，哪些部分是 MLP。
3. 点击某个 XLA 操作，我们可以看到它来自代码中的哪个位置（这对理解 trace 很有用），还能看到指向 Graph Viewer 的链接。

<p markdown=1 class="takeaway">**提示：** 你可以用“电子游戏”风格的按键来浏览 Trace Viewer：A/D 向左/向右平移，W/S 放大/缩小。这些按键让浏览变得容易得多。</p>

### 如何阅读 XLA 操作 {#how-to-read-an-xla-op}

HLO 其实并不难读，而且它对于理解上面 trace 中某一部分对应什么非常有帮助。下面是一个名为 fusion.3 的操作示例。

```c
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

我们把它拆成几个部分来看。

* **操作名称（Op Name）**：fusion.3
  * dot 或 fusion 操作是一组操作，其中最多包含 1 个矩阵乘法，可能还有一系列相关的逐元素 VPU 操作。
* **形状（Shape）**：`bf16[32,32,4096]`
  * 这是该操作的输出形状。我们可以看到 dtype 是 bf16（每个元素 2 字节），`[32,32,4096]` 是形状。
* **布局（Layout）：** `{2,1,0:T(8,128)(2,1)}`
  * `{2,1,0:T(8,128)(2,1)}` 告诉我们各个轴在内存中的顺序（列主序、行主序等）以及数组的填充（padding）。下面会详细介绍。
* **内存位置（Memory location）：** S(1)
  * S(1) 告诉我们这个数组位于 VMEM 中。S(0)（有时会省略）表示 HBM。S(2) 和 S(3) 是其他内存空间。
* **参数（Arguments）**：`bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`
  * 这个操作有一个输入，是一个名为 fusion.32 的 bf16 数组，具有特定的形状。这告诉我们是哪个函数的输出作为了这个函数的输入。

我们再深入理解一下这种表示法。以下面这个简单的例子为例：

`f32[3,5]{1,0:T(2,2)}`

它同样告诉我们，这个操作返回一个形状为 `[3, 5]` 的 float32 数组，具有特定的分块（tiling）`{1,0:T(2,2)}`。虽然分块并不是*特别*重要，但简单来说，分块告诉我们一个 N 维数组是如何在内存中顺序排布的。下图展示了这个数组的排布方式：

{% include figure.liquid path="assets/img/tiling.png" class="img-fluid" %}

在 `{1,0:T(2,2)}` 中，`1,0` 这部分告诉我们数组各维度在物理内存中的顺序，从最次要（most minor）到最主要（most major）。你可以从右往左读这部分，并在 `f32[3,5]` 中找出对应的维度，从而弄清数组的物理布局。在这个例子中，物理布局是 `[3,5]`，与逻辑形状相同。
之后，`T(2,2)` 告诉我们数组被分成 `(2, 2)` 大小的块，在每个块内部，数组先按行排列（**行主序**），再按列排列，即 `(0, 0)` 之后是 `(0, 1)`，然后是 `(1, 0)` 和 `(1, 1)`。由于 `T(2, 2)` 分块，数组被填充到 `[4, 6]`，内存占用扩大了约 1.6 倍。对于上面给出的那个较大的 bf16 数组 `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`，我们使用的是 `T(8,128)(2,1)`，这告诉我们该数组有两级分块：外层是 `(8, 128)` 分块，在这个单元内部还有一个内层的 `(2, 1)` 分块（用于 bf16，使我们的加载总是 4 字节的倍数）。例如，下面是 `bf16[4,8]{1,0:T(2,4)(2,1)}`（不同颜色表示 (2,4) 块，红框表示 (2,1) 块）：

{% include figure.liquid path="assets/img/tiling2.png" class="img-fluid img-small" %}

分块会影响张量的数据块加载到 VMEM 中的效率，XLA 有时会在程序内部引入对张量进行“重新分块”（retile）或“重新布局”（re-layout）的拷贝，有时开销不小。<d-footnote>JAX 提供了一个<a href="https://docs.jax.dev/en/latest/notebooks/layout.html">实验性功能</a>来绕过这个问题，它允许 XLA 为程序的输入计算其“偏好”的布局。当你用 `jax.jit` 对程序进行“即时”（just-in-time）编译时，通常会传入“模拟”输入，告诉 JAX 应预期什么样的形状和 dtype。这些输入通常也携带了分块信息，而这些信息可能不是最优的。你可以改为将输入布局指定为 AUTO，`jax.jit` 就会返回该 jit 程序偏好的布局。然后你可以显式地以该布局加载张量，从而避免在程序内部引入拷贝。</d-footnote>

### Graph Viewer {#graph-viewer}

虽然上面的一些融合看起来可能很复杂，但 XLA Graph Viewer 能让它们更容易解析。例如，下面是一个相当复杂的融合的视图：

{% include figure.liquid path="assets/img/graph-viewer.png" class="img-fluid" %}

多盯着一些 HLO 图看，并尝试把 HLO 操作对应到你正在分析的代码上，这非常有帮助。将鼠标悬停在某个方框上，你通常会看到定义该函数的那行代码。

### 看一个（近乎）真实的性能分析示例 {#looking-at-a-realish-example-profile}

[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 包含一个虚构 Transformer 的性能分析示例。如果你赶时间，[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)有一个 Perfetto 链接，至少可以看到 Trace Viewer。我比平时多花了些功夫，用 `jax.named_scope` 调用来标注这份 trace，好让你能识别出正在发生什么。

{% include figure.liquid path="assets/img/transformer-xprof.png" class="img-fluid" %}

看一看这份性能分析结果，试着真正理解每一部分在做什么。我们先从 FFW 块开始，稍微拆解一下：

{% include figure.liquid path="assets/img/transformer-ffw.png" class="img-fluid" %}

这里我们放大到了 FFW 块。你会看到上投影（up-projection）操作是一个融合（矩阵乘法），输入为 `bf16[8, 1024, 8192]` 和 `bf16[8192, 16384]`，输出为 `bf16[8, 1024, 16384]`。我知道（因为这段代码是我写的）这是一个 4 路数据并行、2 路模型并行分片的矩阵乘法的局部视图，所以我们实际上在做的是

**X:** `bf16[32, 1024, 8192]` \* **W<sub>in</sub>**: `bf16[8192, 32768]` -> **Tmp**: `bf16[32, 1024, 32768]`

**我们预期这需要多长时间？** 首先，每个数据并行分片上的批大小是 `8 * 1024 = 8192`，所以我们应该稳稳地处于计算受限状态。这是在 8 个 TPU v2 核心上运行的，所以我们预期它大约需要 `2 * 32 * 1024 * 8192 * 32768 / (23e12 * 8) = 95.6ms`，这与它实际的耗时（96ms）几乎完全一致。太棒了！这说明我们获得了极好的 FLOPs 利用率！

注意，Google Colab 已经不再提供 TPU v2-8 切片了。要获得一个真正的 8 核切片来跟着操作，你可以使用 [Kaggle](https://www.kaggle.com/)（它仍然免费提供这类切片），或者在 GCP 上申请一个 8 核切片。<d-footnote>如果你只是想在虚构的问题上玩一玩分片，也可以通过 `import jax; jax.config.update("jax_num_cpu_devices", 8)`（大约需要 jax >= 0.4.27）在 CPU 上模拟 8 个设备，然后运行 `print(jax.devices())`。这只适用于玩具问题，不能反映真实的性能。</d-footnote>

**那通信呢？** 你会注意到第二个矩阵乘法末尾藏着一个小小的融合。如果点击它，你会看到

```c
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

它基本上就是一个小型的 ReduceScatter（这是它在 Graph Viewer 中的样子）；

{% include figure.liquid path="assets/img/reduce-scatter-xprof.png" class="img-fluid" %}

我们预期这需要多长时间？嗯，我们是在一个 TPU v2 4x2 上做 ReduceScatter，这应该只需要在 1.2e11 的双向带宽上通信一跳。数组大小为 `2*32*1024*8192`，批次轴被 4 路分片，所以每个分片是 `2*8*1024*8192=128MB`。因此这大约需要 1.1ms。**它实际花了多长时间？** 性能分析结果中报告的是 1.13ms。所以我们已经非常接近 Roofline 了！

**我们也来看看注意力！** 下面是注意力部分的性能分析结果：

{% include figure.liquid path="assets/img/attn-xprof.png" class="img-fluid" %}

我点击了 Q 投影操作，它使用的矩阵 $$W_Q$$ 形状为 [d<sub>model</sub> = 8192, n<sub>heads</sub> = 32, d<sub>qkv</sub> = 256]。我们沿着头（head）维度进行 Megatron 分片。试着做同样的练习，计算这些操作应该花多长时间。

### Memory Profile {#memory-profile}

Memory Profile 让你可以很方便地查看程序内存随时间的变化。这对于调试 OOM 很有帮助。你可以在这里看到，大约有 7.5GB 分配给了模型参数，还有大约 8.5GB 空闲。所以我们还能往内存里放下多得多的东西。

{% include figure.liquid path="assets/img/memory-viewer.png" class="img-fluid" %}

## 练习题 {#worked-problems}

**问题 1**：看一看[这个](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) Colab / 性能分析结果，找出哪里看起来可疑，以及这里发生了什么。你能准确地告诉我正在进行哪些计算、每个操作在做什么吗？所涉及的每个矩阵的真实形状是什么，它们是如何分片的？*试着先只看性能分析结果，不要读代码。*

{% include figure.liquid path="assets/img/all-reduce-profile.png" class="img-fluid" %}

{% details 点击此处查看答案。 %}

这是两个矩阵乘法，具体来说就是：

```py
def matmul(w1, w2, x):
  return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

你可以看到一个 reduce、两个大的融合，以及一个 all-reduce。第一个大的融合是：

```%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1```

它告诉我们每个分片上的形状是 `bf16[8192] * bf16[4096, 8192] -> bf16[4096]`（在 8192 这一维上进行收缩）。通过观察最后那个带有 {% raw %}`replica_groups={{0,16,32,48,64,80,96,112}, ...}`{% endraw %} 的 AllReduce，我们可以看出这是在做 8 路模型并行，所以真实形状是 `bf16[8, 8192] * bf16[32768, 8192] -> bf16[8, 32768]`。

{% enddetails %}

**问题 2：** [前面提到的 Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 实现了一个简单的模拟 Transformer。由于 Colab 已不再提供 TPU v2-8 切片，你需要在 [Kaggle](https://www.kaggle.com/) 或一个 8 核 GCP 切片上运行它来跟着操作。按照 Colab 中的说明，对使用 GSPMD 分区的朴素 Transformer 做一次基准测试。每个部分花了多长时间？应该花多长时间？用的是什么分片方式？试着修复分片！*提示：使用 `jax.lax.with_sharding_constraint` 来约束其行为。修复之后，你能得到的最佳 MFU 是多少？*

作为参考，初始版本大约是每层 184ms，优化后的版本是每层 67ms。完成之后，试着盯着性能分析结果看，看看能否仅凭性能分析结果回答以下问题：

- 这是什么分片策略？
- 批大小、$$d_\text{model}$$、$$d_\text{ff}$$ 分别是多少？
- 花在注意力上的时间与花在 MLP 块上的时间分别占多大比例？
- 按 Roofline 估算，每个操作应该占用多大比例的时间？

**注意：** 自从这道题写成以来，XLA 编译器已经变得更好了。初始版本现在大约是每层 90ms，而优化后的版本每层只快了大约 10ms（每层 80ms）。尽管如此，它仍然值得动手玩一玩，看看你能否做得更好。

<h3 markdown=1 class="next-section">第 9 部分到此结束。第 10 部分将深入探讨 JAX 并行，请点击[这里](../jax-stuff)。</h3>
