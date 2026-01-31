---
layout: post
title: "Attention Mechanisms: From MHA to DeepSeek's MLA"
date: 2026-01-28
description: An in-depth analysis of the KV Cache mechanism and the technological evolution from Multi-Head Attention (MHA) to Grouped-Query Attention (GQA), and finally to DeepSeek's revolutionary Multi-Head Latent Attention (MLA).
tags: [KV Cache, Attention, DeepSeek, LLM, Transformer]
categories: [AI Technology]
giscus_comments: true
related_posts: true
toc:
  sidebar: left
---

## Revisiting Key-Value Cache 


1.1 Computational Logic and Redundancy in Autoregressive Generation

The process of generating text in Transformer models is performed token by token. When predicting the $(t+1)$-th token, the model needs to perform attention calculations combining the current token with all context information from the previous $t$ tokens.

In the original attention mechanism, for every position in the sequence, the Query vector ($Q$), Key vector ($K$), and Value vector ($V$) must be calculated. Without caching technology, the model would have to recalculate the $K$ and $V$ matrices for all historical tokens when generating each new token. This would cause inference complexity to grow quadratically with sequence length ($O(N^2)$).

The introduction of KV Cache aims to transform this repetitive computation into storage space: once the $K$ and $V$ of historical tokens are calculated, they are stored in video memory (VRAM). Subsequent steps only need to calculate the $Q$, $K$, and $V$ for the current new token and concatenate them with the cached historical $K$ and $V$. This mechanism successfully reduces the computational complexity of the inference phase from quadratic to linear ($O(N)$), which is the cornerstone of realizing real-time interactive AI.

### 1.2 Linear Expansion of Memory Usage and the "Memory Wall"

Although KV Cache alleviates the computational bottleneck, it shifts the pressure to memory bandwidth and capacity. The size of the KV Cache is directly proportional to the sequence length, model layers, number of attention heads, and head dimension. The calculation formula is typically expressed as:

$$
Memory_{KVCache} = 2 \times BatchSize \times Layers \times Heads \times Dim \times SeqLength \times BytesPerParam
$$

When processing long texts, the memory occupied by KV Cache often exceeds the model weights themselves. For example, when the context reaches 80,000 tokens, even for a medium-sized model, its KV Cache can occupy tens of GB of VRAM.

When VRAM is full, the system is forced to trigger "CPU Spill" (offloading data to system RAM), causing a dramatic increase in data transfer latency over the PCIe bus. Generation speed can plummet from tens of tokens per second to single digits. This precipitous drop in performance is known as the "Memory Wall" phenomenon.

## Chapter 2: Structural Evolution of Attention Mechanisms: From MHA to GQA

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://shreyansh26.github.io/post/2025-11-08_multihead-latent-attention/images/mla.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
---

### 2.1 Multi-Head Attention (MHA): Accuracy Benchmark and Memory Bottleneck

Multi-Head Attention (MHA) is the cornerstone of the Transformer architecture. By slicing the model dimension into multiple independent heads, it allows the model to simultaneously attend to information from different subspaces at different positions. In MHA, each Query Head is equipped with a unique corresponding Key Head and Value Head.

Although this design has an extremely high ceiling for capturing complex semantic relationships, during inference, all $K$ and $V$ vectors for every layer and every head must be fully cached. When scaling models to hundreds of billions of parameters or extending contexts to tens of thousands of tokens, the demand for memory bandwidth far exceeds the physical limits of GPU HBM (High Bandwidth Memory), leading to massive idling of computational units (ALUs) and creating the so-called "memory-bound" bottleneck.

### 2.2 Multi-Query Attention (MQA): Extreme Bandwidth Optimization

To thoroughly break the memory limit, Multi-Query Attention (MQA) adopts an extreme sharing strategy: letting all query heads share the same pair of Key and Value heads. This design can directly reduce the KV Cache memory usage during inference to $1/H$ of MHA (where $H$ is the number of heads).

MQA greatly improves inference throughput and allows for larger batch sizes, but its limitations are also very obvious. Since all attention heads must reference the same key-value pairs, the model's performance in capturing fine-grained semantic differences and long-range complex dependencies drops significantly, and it is also more prone to instability during training. Therefore, MQA is mostly applied in edge computing devices that are extremely sensitive to resources or in specific small models.

### 2.3 Grouped-Query Attention (GQA): The Art of Engineering Trade-offs

Grouped-Query Attention (GQA) is considered the optimal balance point between MHA and MQA. GQA divides query heads into multiple groups, where query heads within each group share the same pair of key/value heads. For example, the Llama 3 70B model has 64 query heads, and every 8 query heads are grouped together to share one KV pair (i.e., GQA-8).

The introduction of GQA brings the following key advantages:
*   **Significant Memory Reduction**: Taking GQA-8 as an example, KV Cache usage is reduced by 87.5% compared to MHA.
*   **Throughput Improvement**: Since the amount of data read from HBM is drastically reduced, inference speed can usually achieve a 1.5x to 2x improvement.
*   **Accuracy Retention**: Studies show that appropriate grouping strategies can achieve inference efficiency close to MQA with almost no loss in model accuracy.

Currently, GQA has become the standard configuration for mainstream large language models (such as Llama 2/3, Mistral, Qwen 2.5, etc.) and is a core technology for solving modern inference workloads.

#### Memory Optimization Formula

GQA/MQA only modifies the number of Key (K) and Value (V) heads, while keeping the number of Query (Q) heads unchanged (to preserve the model's attention representation capability). Therefore, the Heads term in the original KV Cache formula should be replaced with the actual number of K/V heads ($Heads_{KV}$). The optimized formula is as follows:

$
Memory_{KVCache} = 2 \times BatchSize \times Layers \times Heads_{KV} \times Dim \times SeqLength \times BytesPerParam
$

**Memory Reduction Ratio**: It is directly equal to the grouping factor $G$ ($G = \text{Number of Q Heads} / \text{Number of KV Heads}$). In other words, the KV Cache memory footprint is reduced to $1/G$ of its original size, and the memory occupied by model weights remains almost unchanged (only minor adjustments are made to the parameters of attention layers).

### 2.4 Practical Case Study on a 7B Model (Intuitive Demonstration)

Let’s take the 7B model mentioned earlier ($Layers=32$, $Q \ Heads=32$, $Head \ Dimension=64$) as an example, adopting the industry-standard grouping factor $G=8$ (8 Q heads share one set of K/V heads):

*   **Original MHA**: $Heads_{KV}=32$, KV Cache memory footprint for 80,000 tokens ≈ 640 GB
*   **GQA ($G=8$)**: $Heads_{KV}=32/8=4$, KV Cache memory footprint for 80,000 tokens ≈ 640 GB / 8 = 80 GB (a reduction of 87.5%)
*   **MQA (Extreme Optimization)**: $Heads_{KV}=1$, KV Cache memory footprint for 80,000 tokens ≈ 640 GB / 32 = 20 GB (a reduction of 93.75%)

**Change in Total Inference Memory Footprint**: With GQA enabled, the total inference memory for processing 80,000 tokens is approximately 14 GB (model weights) + 80 GB (KV Cache) + 5 GB (temporary computation memory) ≈ 99 GB. This workload can be handled by only 2 A100 GPUs (80 GB each). Compared to the original requirement of 9 A100 GPUs, the hardware cost is reduced by over 70%.

## Chapter 3: Multi-Head Latent Attention (MLA): DeepSeek's Revolutionary Innovation

### 3.1 Core Logic of Latent Space Compression

Multi-Head Latent Attention (MLA) is one of the most breakthrough technologies introduced in models like DeepSeek-V2/V3. Unlike GQA, which compresses the cache by reducing the number of heads, MLA utilizes the principle of Low-Rank Factorization in linear algebra to compress the $K$ and $V$ vectors of all heads into a compact Latent Space.

In MLA, input vectors are first projected into a latent vector $c_{KV}$ with a rank much smaller than the original model dimension. During the inference phase, the $K$ and $V$ for each head are no longer stored; instead, only this compressed latent vector is stored. When attention calculation is needed, the model dynamically expands the latent vector into the full-dimension keys and values required by each head through an Up-projection Matrix. This method allows DeepSeek models to maintain a very large number of attention heads (e.g., 128 heads) while reducing the memory overhead of KV Cache to about 4.3% to 6.7% of MHA.

### 3.2 Decoupled Rotary Position Embedding (Decoupled RoPE)

Since Rotary Position Embedding (RoPE) is closely related to token position, it possesses position sensitivity, which mathematically conflicts with the linear transformation properties of low-rank compression. If RoPE is applied directly to the latent vector, the compressed information cannot be correctly deconstructed.

DeepSeek's solution is a "Decoupled Attention Structure": it splits the $K$ and $Q$ vectors into two parts—one is the "content part" carrying semantic information, which applies low-rank compression; the other is the "position part" carrying spatial information, which applies RoPE independently. When calculating attention scores, the results of the content part and the position part are superimposed. This design not only overcomes the compatibility problem between RoPE and compression but also further optimizes KV Cache because only the latent vector of the content part and a small portion of the common position vector need to be cached.

### 3.3 Matrix Absorption and Computational Optimization During Inference

MLA's advantage in inference efficiency is reflected not only in space but also in the optimization of computational logic. During inference, MLA allows for "Matrix Absorption": multiple matrix multiplications (transpose of $Q$ and $K$) that originally needed to be performed can be recombined. Since the Up-projection Matrix is static and independent of the input, it can be pre-merged with the weight matrix. This further reduces the computational overhead of MLA during inference, supporting higher throughput autoregressive decoding.

## Summary: Comparison of Different Mechanisms

| Feature | MHA | GQA | MLA (DeepSeek) |
| :--- | :--- | :--- | :--- |
| **KV Cache Compression Principle** | None (Full Storage) | Head Sharing (Reduced Heads) | Low-Rank Factorization (Dimension Compression) |
| **KV Cache Size** | $2 \cdot H \cdot d_h$ | $2 \cdot G \cdot d_h$ | $d_c + d_R$ (Latent Dimension) |
| **Accuracy Retention** | Best (Benchmark) | High (Slight Loss) | Very High (Better or Equal to MHA) |
| **Context Support** | Limited | Medium | Very Strong (Supports 128K+) |
