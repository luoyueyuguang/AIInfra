<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE 01: ZeRO 显存优化实践

> Author by: 许灿岷

目前**GPU + Pytorch + Megatron + DeepSpeed**是常用的训练超大规模语言模型的训练框架。而微软开发的**DeepSpeed**的核心就是**ZeRO**(Zero Redundancy Optimizer)，它是一种显存优化的**数据并行**(data parallelism，DP)方案。**ZeRO**技术通过消除**数据并行**中的显存冗余，显著降低了训练大模型所需的显存。

本实验将深入探讨 ZeRO 的各级优化技术，通过实际代码演示和分析，理解不同级别的 ZeRO 如何实现显存优化。

📌 **PS**：本 Notebook **仅用于教学目的**，所有 ZeRO 实现均为**单 GPU 上的简化模拟**，**并未使用真实多 GPU 并行或通信原语**（如 `all_reduce`, `reduce_scatter`）。真实 ZeRO 需要分布式训练环境（如 DeepSpeed + 多 GPU），其显存节省效果在 **N 个 GPU 时才体现为 1/N**。本实验通过“人为分片 + 手动释放”来**模拟**分片行为，帮助理解核心思想。



## 1. 模型显存占用分析

在深度学习训练中，显存占用可以分为**Activation**和**Model State**两部分：

**Activation**：
- **中间激活值**（Activations）：在前向传播过程中，神经网络的每一层会产生中间激活值，这些激活值需要再反向传播过程中用来计算梯度。
- **输入数据**（Inputs Data）：批处理中输入数据也占用显存，尤其是当批处理较大时。

**Model State**：

- **优化器状态**（Optimizer States）：是 Optimizer 在进行梯度更新时所需要用到数据。一些优化器(如 Adam)需要存储额外的状态信息，如梯度的移动平均值和平方梯度的移动平均值。例如 SGD 中的 Momentum,即使用混合精度训练时的**Float32 Master Parameters**。
- **模型参数**（Parameters）：模型的可学习权重，如存储在显存中的模型权重和偏置项。
- **梯度**（Gradients）：在反向传播过程中计算得到的梯度，用于更新模型参数。其决定了参数的更新方向。

它们三个简称**OPG**，其中**优化器状态**会占据大约 2 倍参数量的显存空间，这取决于选择的优化器，也是整个训练中占据最大空间的部分。

**ZeRO**则在数据并行的基础上，引入了对冗余**Model States**的优化。使用 ZeRO 后，各个进程之后只保存完整状态的**1/GPUs**，互不重叠，不再存在冗余。相比传统数据并行的简单复制，**ZeRO**通过将模型的**参数**、**梯度** 和 **优化器状态**划分到不同的进程来消除冗余的内存占用，也就引出**ZeRO**的三个不同的级别,分别对应**Model States**不同程度的分割(Partition)：

- **ZeRO-1**： 分割**优化器状态**。
- **ZeRO-2**： 分割**优化器状态**与**梯度**。
- **ZeRO-3**： 分割**优化器状态**、**梯度**与**参数**。

![](./images/Code01ZeRO00.png)


对于使用 Adam 优化器的模型，显存占用可估算为：
```
总显存 = 参数显存 + 梯度显存 + 优化器状态显存 + 激活值显存
参数显存 = 参数量 × 4 字节（FP32）
梯度显存 = 参数量 × 4 字节（FP32）
优化器状态显存 = 参数量 × 16 字节（FP32 Adam）
```

显存占用分析工具


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class MemoryAnalyzer:
    """简化的显存分析工具类（仅用于教学演示）"""

    def __init__(self):
        self.memory_stats = defaultdict(list)
        self.previous_allocated = 0

    def record(self, tag=''):
        """记录当前 GPU 显存使用情况（单位：GB）"""
        torch.cuda.synchronize()  # 确保所有操作完成
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        delta = allocated - self.previous_allocated
        self.previous_allocated = allocated

        self.memory_stats['allocated'].append(allocated)
        self.memory_stats['reserved'].append(reserved)
        self.memory_stats['delta'].append(delta)

        print(f"{tag}: 已分配: {allocated:.2f}GB, 变化: {delta:+.2f}GB")
        return allocated

# 创建测试模型
def create_model(hidden_size=2048, num_layers=12):
    """创建一个简单的全连接模型用于测试"""
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

# 执行显存分析
def analyze_memory(seed=42):
    """执行基础训练流程并记录显存变化"""
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return None

    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    analyzer = MemoryAnalyzer()

    analyzer.record("初始状态")

    model = create_model().cuda()
    analyzer.record("模型创建后")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    analyzer.record("优化器创建后")

    inputs = torch.randn(32, 2048, device='cuda')
    targets = torch.randn(32, 2048, device='cuda')
    analyzer.record("数据加载后")

    outputs = model(inputs)
    loss = F.mse_loss(outputs, targets)
    analyzer.record("前向传播后")

    loss.backward()
    analyzer.record("反向传播后")

    optimizer.step()
    analyzer.record("优化器更新后")

    return analyzer.memory_stats

# 执行分析
memory_stats = analyze_memory()
```

    初始状态: 已分配: 0.00GB, 变化: +0.00GB
    模型创建后: 已分配: 0.19GB, 变化: +0.19GB
    优化器创建后: 已分配: 0.19GB, 变化: +0.00GB
    数据加载后: 已分配: 0.19GB, 变化: +0.00GB
    前向传播后: 已分配: 0.20GB, 变化: +0.01GB
    反向传播后: 已分配: 0.39GB, 变化: +0.19GB
    优化器更新后: 已分配: 0.77GB, 变化: +0.38GB


通过这个分析工具，我们可以清楚地看到在每个训练阶段显存的使用情况变化。在实际的大模型训练中，这些显存占用会成倍增长，凸显了 ZeRO 优化的必要性。

```
初始状态: 已分配: 0.00GB, 变化: +0.00GB
模型创建后: 已分配: 0.13GB, 变化: +0.13GB
优化器创建后: 已分配: 0.13GB, 变化: +0.00GB
数据加载后: 已分配: 0.13GB, 变化: +0.00GB
前向传播后: 已分配: 0.14GB, 变化: +0.01GB
反向传播后: 已分配: 0.27GB, 变化: +0.13GB
优化器更新后: 已分配: 0.52GB, 变化: +0.25GB
```

## 2. ZeRO-1: 优化器状态分片

ZeRO-1 通过将优化器状态分片到多个 GPU 上来减少显存占用。在传统数据并行中，每个 GPU 都保存完整的优化器状态副本，这造成了大量的显存冗余。

ZeRO-1 的核心思想是：每个 GPU 只保存一部分优化器状态，当需要更新参数时，通过集合通信操作获取完整的梯度信息。

数学表达上，对于 Adam 优化器，每个 GPU 原本需要存储：

- 参数：$Θ$
- 梯度：$∇Θ$
- 动量：$m$
- 方差：$v$

ZeRO-1 分片后，每个 GPU 只存储：

- 完整参数：$Θ$
- 完整梯度：$∇Θ$
- 1/N 的动量：$m_i$
- 1/N 的方差：$v_i$

其中 N 是 GPU 数量。

![](./images/Code01ZeRO01.png)

ZeRO-1 优化器状态分片


```python
class Zero1Optimizer:
    """ZeRO-1 模拟实现：将参数分片，每个分片独立优化器（单 GPU 模拟）"""

    def __init__(self, params, optimizer_class=torch.optim.Adam, num_shards=4, **kwargs):
        self.params = list(params)
        self.num_shards = num_shards

        # 将参数均匀分片
        self.shards = []
        shard_size = (len(self.params) + num_shards - 1) // num_shards
        for i in range(0, len(self.params), shard_size):
            self.shards.append(self.params[i:i + shard_size])

        # 为每个分片创建独立优化器
        self.optimizers = [
            optimizer_class(shard, **kwargs) for shard in self.shards
        ]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        # 每个分片独立更新（模拟多 GPU 各自更新自己的分片）
        for opt in self.optimizers:
            opt.step()

# 测试 ZeRO-1 效果
def test_zero1(seed=42):
    if not torch.cuda.is_available():
        return None

    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    analyzer = MemoryAnalyzer()

    model = create_model().cuda()
    analyzer.record("模型创建后")

    optimizer = Zero1Optimizer(model.parameters(), num_shards=4, lr=1e-3)
    analyzer.record("ZeRO-1 优化器创建后")

    inputs = torch.randn(32, 2048, device='cuda')
    outputs = model(inputs)
    loss = F.mse_loss(outputs, torch.randn_like(outputs))
    loss.backward()
    optimizer.step()
    analyzer.record("训练一步后")

    return analyzer.memory_stats

# 执行测试
zero1_stats = test_zero1()
```

    模型创建后: 已分配: 0.20GB, 变化: +0.20GB
    ZeRO-1 优化器创建后: 已分配: 0.20GB, 变化: +0.00GB
    训练一步后: 已分配: 0.77GB, 变化: +0.56GB


这个简化实现展示了 ZeRO-1 的核心思想：每个 GPU 只存储和更新一部分参数的优化器状态，通过通信操作确保所有 GPU 的参数保持一致。

```
模型创建后: 已分配: 0.14GB, 变化: +0.14GB
ZeRO-1 优化器创建后: 已分配: 0.14GB, 变化: +0.00GB
训练一步后: 已分配: 0.52GB, 变化: +0.38GB
```

## 3. ZeRO-2: 梯度分片

ZeRO-2 在 ZeRO-1 的基础上进一步优化，不仅分片优化器状态，还分片梯度。这进一步减少了显存占用，因为梯度通常与参数大小相同。

![](./images/Code01ZeRO02.png)

在反向传播过程中，每个 GPU 计算其分配到的参数的梯度，然后通过 Reduce-Scatter 操作聚合梯度。这样每个 GPU 只保存一部分梯度，而不是全部梯度。梯度分片的数学表达：

- 传统方法：每个 GPU 存储完整梯度 $∇Θ$
- ZeRO-2：每个 GPU 存储 1/N 的梯度 $∇Θ_i$


```python
class Zero2Optimizer(Zero1Optimizer):
    """ZeRO-2 模拟：在 ZeRO-1 基础上，只保留当前分片的梯度"""

    def step(self):
        current_shard_idx = 0  # 假设当前 GPU 负责第 0 个分片

        # 删除非本分片的梯度（模拟 reduce-scatter 后释放）
        for shard_idx, shard in enumerate(self.shards):
            for param in shard:
                if param.grad is not None:
                    if shard_idx != current_shard_idx:
                        param.grad = None  # ✅ 释放梯度显存

        # 仅更新本分片
        self.optimizers[current_shard_idx].step()

# 测试 ZeRO-2 效果
def test_zero2(seed=42):
    if not torch.cuda.is_available():
        return None

    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    analyzer = MemoryAnalyzer()

    model = create_model().cuda()
    analyzer.record("模型创建后")

    optimizer = Zero2Optimizer(model.parameters(), num_shards=4, lr=1e-3)
    analyzer.record("ZeRO-2 优化器创建后")

    inputs = torch.randn(32, 2048, device='cuda')
    outputs = model(inputs)
    loss = F.mse_loss(outputs, torch.randn_like(outputs))
    loss.backward()
    optimizer.step()
    analyzer.record("训练一步后")

    return analyzer.memory_stats

zero2_stats = test_zero2()
```

    模型创建后: 已分配: 0.20GB, 变化: +0.20GB
    ZeRO-2 优化器创建后: 已分配: 0.20GB, 变化: +0.00GB
    训练一步后: 已分配: 0.34GB, 变化: +0.14GB


ZeRO-2 通过梯度分片进一步减少了显存占用，但增加了通信开销。在实际应用中，需要根据网络带宽和计算能力权衡这种权衡。

```
模型创建后: 已分配: 0.13GB, 变化: +0.13GB
ZeRO-2 优化器创建后: 已分配: 0.13GB, 变化: +0.00GB
训练一步后: 已分配: 0.31GB, 变化: +0.18GB
```

## 4. ZeRO-3: 参数分片

ZeRO-3 是 ZeRO 系列的最终形态，它不仅分片优化器状态和梯度，还分片模型参数本身。这意味着每个 GPU 只存储模型的一小部分参数，大大降低了单个 GPU 的显存需求。

![](./images/Code01ZeRO03.png)

ZeRO-3 的工作原理：

1. 前向传播时，每个 GPU 只计算它拥有的参数部分
2. 需要其他 GPU 的参数时，通过通信操作获取
3. 反向传播时类似，只计算本地参数的梯度
4. 通过精心设计的通信模式最小化通信开销

参数分片的数学表达：

- 传统方法：每个 GPU 存储完整参数 $Θ$
- ZeRO-3：每个 GPU 存储 1/N 的参数 $Θ_i$


```python
class Zero3Model(nn.Module):
    """ZeRO-3 模拟：仅加载模型的一部分层（参数分片）"""

    def __init__(self, base_model, shard_id=0, num_shards=4):
        super().__init__()
        self.shard_id = shard_id
        self.num_shards = num_shards

        # 计算当前分片负责的层范围
        total_layers = len(base_model)
        layers_per_shard = (total_layers + num_shards - 1) // num_shards
        start = shard_id * layers_per_shard
        end = min(start + layers_per_shard, total_layers)

        # 仅保留本分片的层
        self.layers = nn.ModuleList([base_model[i] for i in range(start, end)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 测试 ZeRO-3 效果
def test_zero3(seed=42):
    if not torch.cuda.is_available():
        return None

    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    analyzer = MemoryAnalyzer()

    base_model = create_model()
    model = Zero3Model(base_model, shard_id=0, num_shards=4).cuda()
    analyzer.record("ZeRO-3 模型创建后")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    analyzer.record("优化器创建后")

    inputs = torch.randn(32, 2048, device='cuda')
    outputs = model(inputs)
    loss = F.mse_loss(outputs, torch.randn_like(outputs))
    loss.backward()
    optimizer.step()
    analyzer.record("训练一步后")

    return analyzer.memory_stats

# 执行测试
zero3_stats = test_zero3()
```

    ZeRO-3 模型创建后: 已分配: 0.06GB, 变化: +0.06GB
    优化器创建后: 已分配: 0.06GB, 变化: +0.00GB
    训练一步后: 已分配: 0.20GB, 变化: +0.14GB


ZeRO-3 提供了最大的显存节省，但通信开销也最大。在实际应用中，通常需要结合各种优化技术，如通信计算重叠、梯度累积等，来平衡显存节省和训练速度。

```
ZeRO-3 模型创建后: 已分配: 0.03GB, 变化: +0.03GB
优化器创建后: 已分配: 0.03GB, 变化: +0.00GB
训练一步后: 已分配: 0.11GB, 变化: +0.08GB
```

## 5. Zero Offload 技术

Zero Offload 技术将优化器状态、梯度和参数卸载到 CPU 内存或 NVMe 存储，进一步扩展了可训练的模型规模。这种技术特别适合在有限 GPU 内存环境下训练超大模型。

![](./images/Code01ZeRO04.png)

Offload 的核心思想是利用 CPU 内存和 NVMe 存储作为 GPU 显存的扩展，通过异步数据传输和计算重叠来最小化性能影响。


```python
class CPUOffloadOptimizer:
    """CPU Offload 模拟：优化器状态存储在 CPU"""

    def __init__(self, params, optimizer_class=torch.optim.Adam, **kwargs):
        self.gpu_params = list(params)
        # 在 CPU 上创建参数副本（无梯度）
        self.cpu_params = [p.detach().cpu().clone() for p in self.gpu_params]
        self.optimizer = optimizer_class(self.cpu_params, **kwargs)

    def step(self):
        # 梯度从 GPU → CPU
        for gpu_p, cpu_p in zip(self.gpu_params, self.cpu_params):
            if gpu_p.grad is not None:
                cpu_p.grad = gpu_p.grad.cpu()

        # 在 CPU 上更新
        self.optimizer.step()

        # 参数从 CPU → GPU
        for gpu_p, cpu_p in zip(self.gpu_params, self.cpu_params):
            gpu_p.data.copy_(cpu_p.data)

        # 清理 CPU 梯度
        for cpu_p in self.cpu_params:
            cpu_p.grad = None

# 测试 CPU Offload 效果
def test_cpu_offload(seed=42):
    if not torch.cuda.is_available():
        return None

    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    analyzer = MemoryAnalyzer()

    model = create_model().cuda()
    analyzer.record("模型创建后")

    optimizer = CPUOffloadOptimizer(model.parameters(), lr=1e-3)
    analyzer.record("CPU Offload 优化器创建后")

    inputs = torch.randn(32, 2048, device='cuda')
    outputs = model(inputs)
    loss = F.mse_loss(outputs, torch.randn_like(outputs))
    loss.backward()
    optimizer.step()
    analyzer.record("训练一步后")

    return analyzer.memory_stats

# 执行测试
offload_stats = test_cpu_offload()
```

    模型创建后: 已分配: 0.20GB, 变化: +0.20GB
    CPU Offload 优化器创建后: 已分配: 0.20GB, 变化: +0.00GB
    训练一步后: 已分配: 0.39GB, 变化: +0.19GB


```
模型创建后: 已分配: 0.13GB, 变化: +0.13GB
CPU Offload 优化器创建后: 已分配: 0.13GB, 变化: +0.00GB
训练一步后: 已分配: 0.25GB, 变化: +0.12GB
```

## 6. 性能分析与实验结果

为了验证 ZeRO 各级别的效果，我们设计了以下实验：


```python
# 汇总所有方法的显存使用情况
def compare_methods():
    if not torch.cuda.is_available():
        return

    print("\n 显存使用对比 (单位: GB):")
    print("-" * 40)

    # 重新运行基础测试
    baseline = analyze_memory()
    print("-" * 40)
    zero1 = test_zero1()
    print("-" * 40)
    zero2 = test_zero2()
    print("-" * 40)
    zero3 = test_zero3()
    print("-" * 40)
    offload = test_cpu_offload()
    print("-" * 40)

    # 提取最终显存使用量
    print(f"基础方法: {baseline['allocated'][-1]:.2f}GB")
    print(f"ZeRO-1: {zero1['allocated'][-1]:.2f}GB ({(1-zero1['allocated'][-1]/baseline['allocated'][-1])*100:.1f}% 节省)")
    print(f"ZeRO-2: {zero2['allocated'][-1]:.2f}GB ({(1-zero2['allocated'][-1]/baseline['allocated'][-1])*100:.1f}% 节省)")
    print(f"ZeRO-3: {zero3['allocated'][-1]:.2f}GB ({(1-zero3['allocated'][-1]/baseline['allocated'][-1])*100:.1f}% 节省)")
    print(f"CPU Offload: {offload['allocated'][-1]:.2f}GB ({(1-offload['allocated'][-1]/baseline['allocated'][-1])*100:.1f}% 节省)")

# 执行对比
compare_methods()
```

    
     显存使用对比 (单位: GB):
    ----------------------------------------
    初始状态: 已分配: 0.02GB, 变化: +0.02GB
    模型创建后: 已分配: 0.20GB, 变化: +0.19GB
    优化器创建后: 已分配: 0.20GB, 变化: +0.00GB
    数据加载后: 已分配: 0.20GB, 变化: +0.00GB
    前向传播后: 已分配: 0.21GB, 变化: +0.00GB
    反向传播后: 已分配: 0.39GB, 变化: +0.18GB
    优化器更新后: 已分配: 0.77GB, 变化: +0.38GB
    ----------------------------------------
    模型创建后: 已分配: 0.20GB, 变化: +0.20GB
    ZeRO-1 优化器创建后: 已分配: 0.20GB, 变化: +0.00GB
    训练一步后: 已分配: 0.77GB, 变化: +0.56GB
    ----------------------------------------
    模型创建后: 已分配: 0.20GB, 变化: +0.20GB
    ZeRO-2 优化器创建后: 已分配: 0.20GB, 变化: +0.00GB
    训练一步后: 已分配: 0.34GB, 变化: +0.14GB
    ----------------------------------------
    ZeRO-3 模型创建后: 已分配: 0.06GB, 变化: +0.06GB
    优化器创建后: 已分配: 0.06GB, 变化: +0.00GB
    训练一步后: 已分配: 0.20GB, 变化: +0.14GB
    ----------------------------------------
    模型创建后: 已分配: 0.20GB, 变化: +0.20GB
    CPU Offload 优化器创建后: 已分配: 0.20GB, 变化: +0.00GB
    训练一步后: 已分配: 0.39GB, 变化: +0.19GB
    ----------------------------------------
    基础方法: 0.77GB
    ZeRO-1: 0.77GB (0.0% 节省)
    ZeRO-2: 0.34GB (55.0% 节省)
    ZeRO-3: 0.20GB (73.4% 节省)
    CPU Offload: 0.39GB (48.9% 节省)


通过这个实验，我们可以清楚地看到 ZeRO 各级别对显存占用的优化效果。在实际的大模型训练中，这些优化可以带来数倍甚至数十倍的显存节省。

```
显存使用对比 (单位: GB):
----------------------------------------
基础方法: 0.39GB
ZeRO-1: 0.39GB (0.0% 节省)
ZeRO-2: 0.31GB (20.5% 节省)
ZeRO-3: 0.11GB (71.8% 节省)
CPU Offload: 0.25GB (35.9% 节省)
```

## 总结与思考

ZeRO 技术通过分片优化器状态、梯度和参数，显著降低了大模型训练的显存需求。本实验通过代码实现和原理分析，深入探讨了：

1. **ZeRO-1**：优化器状态分片，减少约 4 倍显存占用
2. **ZeRO-2**：梯度分片，进一步减少约 8 倍显存占用  
3. **ZeRO-3**：参数分片，最大可减少约 N 倍显存占用（N 为 GPU 数量）
4. **Zero Offload**：将数据卸载到 CPU/NVMe，支持训练超大模型

这些技术可以组合使用，根据具体的硬件环境和模型大小选择最合适的配置。在实际应用中，DeepSpeed 框架提供了完整的 ZeRO 实现，建议直接使用经过优化的官方实现。

## 引用与参考

- https://arxiv.org/abs/1910.02054
- https://www.cnblogs.com/whiteBear/p/18341975
