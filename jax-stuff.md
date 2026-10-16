---
layout: distill
title: "用 JAX 为 TPU 编程"
# permalink: /main/
description: "如何用 JAX 高效地为 TPU 编程！本节的大部分内容取自<a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>这里</a>。你可以在 <a href='https://colab.sandbox.google.com/'>Google Colab</a> 上使用免费的 TPU 运行本节中的代码示例。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 10

previous_section_url: "../profiling"
previous_section_name: "第 9 部分：性能分析"

next_section_url: ../conclusion
next_section_name: "第 11 部分：结论"

giscus_comments: false
giscus_discussion: 12

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
  - name: Yash Katariya
    url: https://x.com/yashk2810
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "JAX 中的并行是如何工作的？"
    id: how-does-parallelism-work-in-jax
  - subsections:
    - name: "自动分片模式"
      id: auto-sharding-mode
    - name: "显式分片模式"
      id: explicit-sharding-mode
    - name: "通过 shard_map 实现的手动分片模式"
      id: manual-sharding-mode-via-shard-map
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

## JAX 中的并行是如何工作的？ {#how-does-parallelism-work-in-jax}

对于多设备编程，JAX 支持三种思路：

1. **编译器，你来掌舵！** 让 XLA 编译器自动对数组进行划分，并决定需要添加哪些通信来实现给定的程序。这样，你可以把一个在单个设备上运行的程序，不做任何修改就自动运行在数千个设备上。
2. **JAX，你来掌舵！** 自动并行很好，但有时编译器会做出一些离谱的事情。显式分片让你像往常一样编写单设备代码，但由 JAX（而不是编译器）来处理分片传播。这意味着当你的意图不明确时，JAX 可以要求你加以澄清。
3. **就让我按我的意思写，该死的！** 编译器固然不错，但它们有时会做错事，添加你并不想要的通信。有时我们希望明确指定究竟要执行哪些通信。

| 模式 | 视角？ | 显式分片？ | 显式集合通信？ |
|:---:|:---:|:---:|:---:|
| 自动（Auto） | 全局 | ❌ | ❌ |
| 显式（Explicit） | 全局 | ✅ | ❌ |
| 手动（Manual） | 每设备 | ✅ | ✅ |

相应地，JAX 为每种模式都提供了 API：

1. `jax.jit`（配合 `Auto` 网格轴）让你可以拿任何现有的 JAX 函数，用已分片的输入来调用它。JAX 随后会使用 XLA 的 [Shardy](https://openxla.org/shardy) 编译器自动并行化该程序。XLA 会在需要时为你添加通信（AllGather、ReduceScatter、AllReduce 等），以支持现有的操作。虽然它并不完美，但通常能很好地把你的程序自动扩展到任意数量的芯片上，而无需修改代码。
2. 配合 `Explicit` 网格轴的 `jax.jit` 看起来与 (1) 类似，但由 JAX 而不是 XLA 来处理分片传播。这意味着数组的分片实际上是 JAX 类型系统的一部分，当 JAX 检测到有歧义的通信时，它可以报错并让用户来解决。
3. `jax.shard_map` 是更手动的对应方案。你得到的是程序的设备本地视角，必须显式写出你想要的所有通信。有一个已分片的数组，并想在每个设备上都拿到完整的数组？添加一个 `jax.lax.all_gather`。想在所有设备间对一个数组求和？添加一个 `jax.lax.psum`（即 AllReduce）。这样编程更难，但做出你不想要的事情的可能性要小得多。

<h3 id="auto-sharding-mode">自动分片模式</h3>

`jax.jit` 在 JAX 中扮演两个角色。顾名思义，它把一个 Python 函数“即时”（just-in-time）编译成字节码（经由 XLA/HLO/LLO），从而运行得更快。但如果输入是已分片的，或者用户指定了 `in_sharding` 或 `out_sharding`，它还会让 XLA 把计算分布到多个设备上，并按需添加通信。例如，下面展示了如何用 `jax.jit` 编写一个分片的矩阵乘法：

```py
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

# 这会创建一组虚拟的 8 个 CPU 设备，这样你就可以在没有 TPU 的 CPU 上运行这段代码。
jax.config.update("jax_num_cpu_devices", 8)

# 这会创建一个 4x2 的二维 mesh，轴名为 X 和 Y，JAX 会默认使用它。
# 我们显式地告诉 JAX，让 XLA 编译器沿这些轴推断分片。
mesh = jax.make_mesh(axis_sizes=(4, 2), axis_names=('X', 'Y'), axis_types=(Auto, Auto))
jax.set_mesh(mesh)

# 我们创建一个矩阵 W 和输入激活值 In，它们分片在我们的各个设备上。
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# 我们可以在这里显式编译这个分片的 matmul 函数。这会加入所有
# 必要的通信（例如 matmul 之后的一个 AllReduce）。
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

这段代码可以在任意分片下自动运行，并把计算划分到我们的各个设备上。**但在硬件层面究竟发生了什么？**

1. 首先，我们创建在各设备间分片的 In 和 W<d-footnote>注意我们是怎么做的。这是创建具有特定分片的数组的一种方式（即在创建函数中加入 device 参数）。另一种方式是先用 `jnp.array(....)` 正常创建数组，然后再执行例如 `jax.device_put(..., jax.P('X', 'Y'))`。还有一种方式是编写一个创建所需数组的函数，并在对它做 jit 编译时把 `out_shardings` 设为你想要的分片。</d-footnote>。W 沿收缩维度做了 2 路分片，而 In 做了 8 路分片：沿输入维度 4 路、沿收缩维度 2 路。这对应于分片 W[D<sub>Y</sub>, F] 和 In[B<sub>X</sub>, D<sub>Y</sub>]，也就是模型并行与数据并行的一种组合。
2. 如果我们在本地（即在单个设备上）运行它，`matmul_square` 只会对输入求平方，然后执行一次简单的矩阵乘法。但因为我们把 `out_shardings` 指定为 `P('X', None)`，输出将沿批次维度分片、在模型维度上复制，因此需要一次 AllReduce 来计算。

用前面章节中的记号，它很可能会执行类似下面的操作：

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit` 会自动为我们添加这些！我们实际上可以用 `jit_matmul.as_text()` 打印出 HLO，看到如下 HLO（做了大幅删节）：

```py
# 这个 fusion 就是分片输入与矩阵之间实际的 matmul
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# 我们在各设备之间对部分求和的结果做归约
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

我们可以在上面看到矩阵乘法（即那个 fusion）和 AllReduce。尤其要注意各个形状。`bf16[2, 1024]` 是激活值的本地视图，因为我们的 `batch_size=8` 被切分到 4 个设备上，而 `d_model=2048` 同样被切分为 2 份。

**这相当神奇！** 无论我们的程序多么复杂，[Shardy](https://openxla.org/shardy) 和 jit 都会尝试为所有中间激活值找到分片方式，并按需添加通信。话虽如此，Shardy 也有它的缺陷。它可能会犯错。有时你查看性能分析结果，会发现哪里出了问题：一个巨大的 AllGather 占据了 80% 的时间，而它本来是不必要的。发生这种情况时，我们可以尝试用 `jax.lax.with_sharding_constraint` 显式标注中间张量来纠正编译器。例如，对于两次矩阵乘法，我可以用下面的代码强制让中间激活值沿 `y` 维度分片（并不是说这是个好主意）：

```py
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X', 'Y'), (Auto, Auto))
jax.set_mesh(mesh)

def matmul(x, W_in, W_out):
  hidden = jnp.einsum('bd,df->bf', x, W_in)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('X', 'Y'))
  return jnp.einsum('bf,df->bd', hidden, W_out)
```

在自动划分的世界里，JAX 并行编程大约有 60% 的工作就是这样：通过 `jax.lax.with_sharding_constraint` 来控制中间分片。但众所周知，“挠编译器痒痒”（compiler tickling）并不是一种令人愉快的编程模型。你可以标注每一个中间变量，却仍然不知道能否得到正确的结果。那么，如果 JAX 本身就能处理并控制分片传播呢？

<h3 id="explicit-sharding-mode">显式分片模式</h3>

显式分片（或称“类型中的分片”，sharding in types）看起来与自动分片很像，但分片传播发生在 JAX 层面！每个 JAX 操作都有一条分片规则，它接收该操作各参数的分片，并为操作的结果产生一个分片。你可以用 `jax.typeof` 查看得到的分片：

```py
import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# 在 TPU v5e 2x2 上运行。这会为硬件的两个物理轴分配名称。
mesh = jax.make_mesh(axis_sizes=(2, 2), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))

# 这告诉 JAX 对所有操作都使用这个 mesh，因此你只需指定 PartitionSpec P 即可。
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16, dtype=np.float32).reshape(8, 2), jax.P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # float32[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # float32[8@X,2@Y]
  return out

f(x)
```

如你所见，JAX 把分片从输入（`x`）传播到了输出（`out`），这些分片在 trace 时可以通过 `jax.typeof` 查看。对于大多数操作，这些规则简单而显然，因为只有一种合理的选择（例如逐元素操作会保持相同的分片）。但对于某些操作，结果应如何分片是有歧义的，这时 JAX 会在 trace 时抛出错误，要求程序员显式提供 `out_sharding` 参数（例如 jnp.einsum、jnp.reshape 等）。我们再看一个存在冲突的例子：

```py
# 我们创建一个矩阵 W 和输入激活值 In，它们分片在我们的各个设备上。
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # 这里会报错
```

这段代码会报如下错误：

```
Contracting dimensions are sharded and it is ambiguous how the output should be sharded.
Please specify the output sharding via the `out_sharding` parameter.
Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)
```

这非常棒，因为 einsum 的输出应该如何分片确实是有歧义的。输出分片可以是：
* P('X', 'Y')，这会引入一次 ReduceScatter，或者
* P('X', None)，这会引入一次 AllReduce

与 Auto 模式不同，显式模式在检测到有歧义的通信时会报错，并要求用户来解决。所以在这里你可以这样做：

```py
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=jax.P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

Auto 模式和 Explicit 模式可以通过 `jax.sharding.auto_axes` 和 `jax.sharding.explicit_axes` API 组合使用。想了解更多信息，这是一篇[非常值得一读的文档](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)。

<h3 id="manual-sharding-mode-via-shard-map">通过 shard_map 实现的手动分片模式</h3>

如果说 Shardy 是“编译器来掌舵”的模式，那么 jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) 则把一切都交到你手里。你像在 jax.jit 中一样指定输入的分片，但随后要显式写出所有通信。`jax.jit` 给你的是程序的全局跨设备视角，而 `shard_map` 给你的是每个设备的本地视角。

下面是一个例子。试着推理一下这个函数做了什么：<d-footnote>如果你想在 colab 中通过模拟一个网格来亲自试验，可以使用下面这个单元格：`import jax; jax.config.update('jax_num_cpu_devices', 8)`</d-footnote>

```py
import jax
import jax.numpy as jnp

Explicit = jax.sharding.AxisType.Explicit

mesh = jax.make_mesh((2, 4), ('x', 'y'), (Explicit, Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=jax.P(('x', 'y')))

# 这个函数将作用于数组的 1/8。
@jax.shard_map(in_specs=jax.P(('x', 'y')), out_specs=jax.P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**它做了什么？** `slice_and_average` 在每个 TPU 上运行，处理数组的 1/8，从中切出前 4 个元素，并在整个网格上对它们求平均。这意味着我们实际上在计算 `mean(x[:4], x[64:68], x[128:132], …)`。这相当酷，因为用其他方式很难在 JAX 中表达这种操作。

**为什么要这样做，而不是用 jax.jit？** 如果我们用了 `jax.jit`，`slice_and_average` 看到的将是数组的全局视图（完整的 `[512,]` 数组）。我们就得切出这个不均匀的切片，再执行求平均，而 XLA 还必须正确理解这一操作。XLA 可能会添加错误的通信，或者被搞糊涂。而在这里，我们看到的是本地视图，只需写出我们需要的通信。

**示例 [集合矩阵乘法（Collective Matmul）]：** 举一个更实际的例子：假设我们要实现模型并行，其中激活值最初是按模型维度分片的，即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]。朴素的做法是先对 A 做 AllGather，然后执行一次本地矩阵乘法：

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

遗憾的是，这样做不好，因为它无法让通信与计算重叠。要让二者重叠，可以使用 [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959) 中描述的“集合矩阵乘法”（collective matmul）。该算法大致如下：

* 对于每个 Y 分片，用 A 的本地块与 W 的本地块做矩阵乘法，得到形状为 `[B / X, F / Y]` 的结果。与此同时，对 A 进行置换（permute），使你在本地拿到下一个块，再执行矩阵乘法并把结果累加起来。

我们可以用 `jax.shard_map` 相当轻松地实现它：

```py
import functools

import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# 这段代码旨在 TPU v5e-8 运行时上运行。如果你无法获得这种运行时，
# 可以尝试设置 jax.config.update('jax_num_cpu_devices', 8)。
#
mesh = jax.make_mesh(axis_sizes=(2, 4), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs 是循环传递的操作数；rhs 是本地操作数
  axis_size = jax.lax.axis_size('Y')  # 在本例中 axis_size = 4
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # 对一个分块做 matmul
    update = lhs @ rhs_chunk
    # 向左循环移位
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pcast(accum, ('X', 'Y'), to='varying')
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # 在最后一次 permute 之后计算最后一个分块，使 lhs 恢复到最初的状态
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

这相当巧妙！我们可以对它做基准测试，会发现它也快了很多！[这里](https://imgur.com/a/e9I6SrM)是使用默认 jit 矩阵乘法时的性能分析结果，耗时 311us，开头有一个很大的阻塞式 AllGather：

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

而[这里](https://imgur.com/a/21iy0Sv)是上面那个版本，耗时 244us。你可以看到性能分析结果中没有了 AllGather，全都是有用的工作！我们的 FLOPs 利用率也高了很多。

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

还值得一提的是，在收缩维度上不做分片时，矩阵乘法耗时为 [224us](https://imgur.com/a/i3gNKfq)，所以我们这里已经非常接近未分片的基线了。这是一个很好的例子，展示了为提高 TPU 利用率你可能最终要做的那类性能工程。想看更多 `shard_map` 的示例，[这篇笔记非常好](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side)。

下面是几道有用的练习题，可以尝试用 `jax.jit` 或 `shard_map` 来实现！

## 练习题 {#worked-problems}

下面是一些与 JAX 相关的随机问题，以后我还会再补充一些。所有这些题目都需要一定数量的 TPU。Colab 已不再提供 TPU v2-8 切片，所以请使用 [Kaggle](https://www.kaggle.com/)（它仍然免费提供）或一个 8 核的 GCP 切片。<d-footnote>如果你只是想在玩具问题上模拟一个网格（mesh），也可以用 `import jax; jax.config.update('jax_num_cpu_devices', 8)` 在 CPU 上伪造 8 个设备（大约需要 jax >= 0.4.27），不过这无法反映真实性能。</d-footnote>从现在开始，我们假设你有 N 个可用设备。

**问题 1：** 设 **A** 是一个形状为 float32[S<sub>X</sub>, D<sub>Y</sub>] 的激活值数组，且 `X * Y = N`。完成以下任务：

1. 用 JAX 写一个函数，计算每个 `(X, Y)` 分片内的平均值，即返回一个大小为 [X, Y] 的数组，其中 `arr[i, j]` 是分片 `(i, j)` 上的平均值。分别用 `jax.jit` 和 `shard_map` 实现。对两者分别做性能分析，看看各自耗时多少。是否引入了任何通信？*提示：本不应该有，但 XLA 有时还是会加上。*

2. 用 JAX 写一个函数，对某个 shift 返回 `roll(x, shift, axis=0) - x`，且 roll **在每个沿 X 的分片内部**进行。我还没有受虐到要让你用 jax.jit 来做这件事，所以只用 `shard_map` 实现即可。

{% details 点击此处查看答案。 %}

**第 1 部分：** 下面是第 1 部分的一个解法。注意 `jax.jit` 解法中我们需要做的相当复杂的 reshape。

```py
import numpy as np

import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

**第 2 部分：** 下面是第 2 部分的类似解法。题目要求的是 `shard_map` 版本，这里也附上了一个 `jax.jit` 版本以便对比。

```py
import numpy as np

import jax
import jax.numpy as jnp

import functools

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0) - x,
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return (jnp.roll(reshaped, shift, axis=1) - reshaped).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

**问题 2：** 这里我们一起来构建一个基础的“混合专家”（mixture of experts）模型。设 **W**: float32[E<sub>X</sub>, D, F] 是一组 E 个“专家”矩阵。设 **A**: float32[S<sub>X</sub>, D]（我们的激活值），并设 **B**: int32[S<sub>X</sub>] 是一组“路由分配”，其中 B[i] 是 `[0, E)` 范围内的整数，告诉我们要用哪个矩阵来处理该激活值。我们想用 JAX 写一个函数，返回 `Out[i] = A[i] @ W[B[i]]`。

1. 我们先完全忽略分片。把所有这些张量都设得足够小，使其能放进单个设备。写出该函数的一个本地实现。*确保不要物化一个形状为 `[S, D, F]` 的数组！提示：有几种方法可以做到。最简单的是遍历各个专家，用 `W[e]` 乘以整个 **A**，再把没有路由到专家 `e` 的 token 屏蔽掉。或者，你可以按专家把 token 排序到一个形状为 `[E, S, D]` 的填充缓冲区中，做一次批量矩阵乘法（matmul），并屏蔽掉填充部分（为什么第二个维度的大小需要是 S？）。`jax.lax.ragged_dot` 实现的就是这种排序版本，且无需填充。*

2. 如果你直接对上面的方法使用 `jax.jit`，会发生一些事情。对它做性能分析，看看它决定执行什么通信。耗时多少？

3. 你会注意到上面方法的一个问题：它很可能会在本地收集完整的激活值 **A**，即 AllGather<sub>X</sub>([S<sub>X</sub>, D])。这不仅在通信上代价高昂，而且如果我们无法在本地放下完整的激活值，在内存上的代价也极其高昂。请使用 `shard_map` 和显式通信来实现上述功能。

      1. 作为第一版，最简单的做法可能是使用 `jax.lax.all_gather`，并像第 1 步那样重新排序。

      2. 作为第二版，尽量避免物化任何大小为 `[E, S, D]` 的数组，即尝试在 `jax.lax.while_loop` 中使用 `jax.lax.all_to_all`，以不规则（ragged）的方式执行计算。这样就可以避免物化完整的激活值，也不会在填充上浪费计算。这比你最初的实现快多少？

4. 大多数 MoE 会路由到多个（k 个）专家，然后对结果取平均。重构上面的实现来支持这一点。这种情况下，令 **B**: int32[S<sub>X</sub>, k] 表示要路由到的 k 个专家。

{% details 点击此处查看（部分）答案。 %}

**第 1 部分：** 这里有很多种选择。下面是最简单的一种，它只是带屏蔽地遍历各个专家，从不对 token 排序：

```py
def moe_local(W: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry  # [S, F]
        mask = (B == e)[:, None]  # [S, 1]
        expert_result = A @ W[e]  # [S, F] - 该专家对所有 token 的变换
        output = output + expert_result * mask  # 只保留分配给该专家的 token 的结果
        return output, None

    output = jnp.zeros((S, F))
    output, _ = jax.lax.scan(expert_forward, output, jnp.arange(E))

    return output
```

这会浪费计算，因为每个专家都要处理每个 token。更高效的做法是按专家对 token 排序，然后使用 `jax.lax.ragged_dot`：它接收排好序的激活值以及每个专家的 token 数量，在没有任何填充或屏蔽的情况下完成矩阵乘法。

**第 3 部分：** 这里我只给出伪代码的大致框架（如果你有简洁的解法，欢迎补充）：

```py
chunk_size = 128
def matmul(W, x, B):
  i = 0
  outs = []
  x = # 根据分配结果对 x 排序
  while (chunk := x[i:i+chunk_size]).any():
     chunk = all_to_all(chunk)
     outs.append(matmul_local(W, chunk))
     i += chunk_size
  return concat(outs)
```

基本思路是遍历数组的各个块，对它们排序并执行 all_to_all，然后执行本地的 FLOPs。

{% enddetails %}

**问题 3：** 上面的集合矩阵乘法（collective matmul）示例其实与真实的 LLM 密切相关。我们来调整一下这个示例，实现完整的 Transformer 层栈。

1. 作为练习，我们先实现一个 AllReduce 集合矩阵乘法，即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]。注意输出只沿 X 分片，即它在 Y 上是复制的。朴素算法上文已经讨论过，基本上就是一次本地矩阵乘法后接一次 AllReduce。尝试为该操作实现一个通信重叠的“集合”版本。*提示：在输出维度上分块，可以随意使用 `jax.lax.psum`（即 AllReduce）。* *注意：由于 XLA 处理这种情况的方式，它实际上可能并不比基线更快。*

2. 与上面的 AllReduce 集合矩阵乘法互补的是 ReduceScatter 集合矩阵乘法，例如 Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]。它出现在 Transformer 的下投影（down-projection）矩阵中。用 JAX 实现它的一个通信重叠的集合版本。注意只传递你所需的最少数据。*提示：尝试在累加结果的同时对其进行置换（permute）。*

3. 把正文中的 AllGather 集合矩阵乘法与 (2) 中的 ReduceScatter 集合矩阵乘法组合成一个端到端的 Transformer 块，以重叠通信的方式执行 In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]。<d-footnote>和之前一样，我们不能先计算 $W_{in} \cdot W_{out}$，因为这里省略了一个非线性操作。</d-footnote>这比 `jax.jit` 实现快多少？

**问题 4：** 上面实现的所有集合矩阵乘法都是单向的：它们只朝一个方向置换。重写 AllReduce 集合矩阵乘法和 ReduceScatter 集合矩阵乘法，使其使用双向通信。它们会快多少？

### 第 10 部分到此结束。基本上就是这些了！最终结论与延伸阅读请点击[这里](../conclusion)。 {#thats-all-for-part-10-thats-basically-it-for-final-conclusions-and-further-reading-click-here}
