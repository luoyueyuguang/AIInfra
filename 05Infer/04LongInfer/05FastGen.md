# FastGen
Author by: 汪袁烁


*提示：如果你对于 Prefill 和 Decode 阶段还不是很了解，可以去看我的`./PD 分离`，我在那里对于 Prefill、Decode 以及大模型推理框架中它们是如何运作的进行了一个简要介绍。*

## Introduction

长提示工作负载变得越来越重要，而现有的 VLLM 等框架面临长提示词难以维持高质量的服务。此外，众所周知，LLM 的推理主要分为`prefill`（用户输入 prompt）和`decode`（LLM 输出 token）两个阶段，然而 `decode`的生成阶段往往会打断`prefill`阶段：

>When systems treat them as distinct
>phases, generation will be preempted by prompt processing that risks breaking the service level
>agreements (SLAs).

更具体的，在多个请求进入之后先进行先来先服务的调度策略，但是一旦显存被打满之后，可能后来的请求会通过`Swapping`策略（先释放被抢占请求的所有 block，然后从 gpu 上交换到 cpu 上）
或者`Recomputation`策略（直接释放它们的物理块，把它们重新放入等待处理的队列中，等后续资源充足时再重新从 prefill 阶段开始做推理）。从而发生打断的情况。

为了解决该痛点，本文的主角，`DeepSpeed-FastGen`诞生了。


## background

你也许会想，这个问题难道没有前人发现或者尝试解决吗？当然，在 FastGen 之前，我们往往采用以下方法来处理该问题：

### 分块 KV 缓存

我们知道`KVCache`用来缓存之前所有已经生成的 token 的 key 和 value 向量，目的是在后续生成 token 时重用这些计算，避免重复的 attention 计算。然而它具有
内存碎片化 / 内存浪费 / 预分配无效空间 等问题。为了处理这些问题，VLLM 团队提出了 PagedAttention。更具体的，它通过将每个请求的 KV cache 分割成固定大小的块等方法，可以灵活处理这些空间碎片。

然而它也有自己的局限性，在长 prompt（上下文很长）的场景下，加载 / 处理这个 prompt 本身就是一个巨大的计算 / 内存操作。如果系统设计是区分 prompt processing 和后续 token generation 阶段，有可能在 prompt 阶段占用很长时间或带来调度不平衡。

关于 PagedAttention 的更多细节，我已经放在了`./05-1PagedAttention.md`上了。


### 连续批处理

之前我们说过，`decode`和`prefill`往往被视为两个阶段，这样往往造成负载不均，并且在面临长文本序列的输入时容易花费较长时间。而“Continuous batching”（连续批处理）则允许 请求 随时加入 / 离开 运行的 batch，而不是在 prompt 阶段与生成阶段严格分割、或者说先把所有 prompt 全部处理完才开始生成。也就是说批次是“动态流式”的，而不是静态预先构造。

## DeepSpeed-FastGen

DeepSpeed-FastGen 的目标是利用连续批处理和非连续 KV 缓存技术，以提升数据中心服务大型语言模型（LLM）的硬件利用率和响应速度。

## DynamicSplitFuse


它的核心思想是：

对较长的 prompt，将其 拆（Split） 成多个小段，在多个 forward pass 中逐步处理，只在最后一段进行生成（token output）。

对较短的 prompt，将多个 prompt 合并 融合（Fuse），填满目标的 token 预算，使得每个 forward pass 的总 token 数稳定且高效。

通过这样 拆 + 合并的组合，使得每次 forward pass 的 token 数比较集中、稳定，从而更可能落在 GPU 吞吐效率较高的区间。

简而言之，Dynamic SplitFuse 在 PagedAttention 的基础之上引入 prompt 分片、prompt + 生成融合的策略，用来保持每个 forward pass 的 token 数保持一个稳定 / 合理的预算，从而让 GPU 在高吞吐率区域运行更稳定，减少由于 prompt 本身过长引起的不平衡、长前向传递导致的延迟 / 抢占问题。




# Ref

1. [DeepSpeed-FastGen 论文](https://arxiv.org/pdf/2401.08671v1)
2. [DeepSpeed-MII 文档](https://deepspeed-mii.readthedocs.io/en/rtd-staging/)
3. [OpenLMAI 博客](https://openlm.ai/deepspeed-fastgen/)
4. [DeepSeed 知乎](https://zhuanlan.zhihu.com/p/665494115)
5. [PagedAttention 论文](https://arxiv.org/pdf/2309.06180)
6. [关于 PD 分离](https://www.bilibili.com/video/BV1wcdbYwE6s?spm_id_from=333.788.videopod.sections&vd_source=8a4545c25a1c1192fb4e7e037876e6f4)

