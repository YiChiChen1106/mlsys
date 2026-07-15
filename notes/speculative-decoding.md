# Speculative Decoding

Speculative decoding is a family of LLM inference acceleration methods for reducing decode latency.

Beginner mental model:

```text
small/fast drafter guesses several future tokens
large target model verifies them in parallel
accepted tokens are committed
rejected tokens fall back to normal target-model generation
```

The goal is to turn the slow one-token-at-a-time decode loop into a process that can sometimes accept multiple tokens per target-model forward pass.

## Why It Exists

LLM decode is autoregressive:

```text
token 1 -> token 2 -> token 3 -> token 4
```

The target model normally generates one token per forward pass. This is slow because every output token requires another model step.

Speculative decoding tries to reduce the number of expensive target-model steps.

## Classic Draft-Then-Verify Flow

Classic speculative decoding uses:

- a large target model,
- a smaller or faster draft model.

Flow:

```text
1. draft model proposes K future tokens
2. target model verifies those K tokens in parallel
3. accept a prefix of correct/speculatively valid tokens
4. if a token is rejected, recover with target-model sampling
5. continue
```

If many draft tokens are accepted, one target-model call can produce multiple output tokens.

This is why the key metric is:

```text
acceptance rate
```

Higher acceptance means more tokens committed per expensive target-model verification.

## Lossless vs Approximate

The original speculative sampling algorithm is designed to preserve the target model's output distribution.

Beginner interpretation:

```text
the drafter proposes
the target model remains the judge
```

So if implemented with the correct acceptance/rejection rule, speculative decoding can accelerate generation without changing the target distribution.

In production, always check the exact implementation and sampling settings before claiming losslessness.

## Why It Can Be Faster

The target model can score several draft tokens in one parallel verification pass.

The bet is:

```text
drafting cost + verification cost
<
normal target model decoding cost for K tokens
```

Speculative decoding is more likely to help when:

- concurrency is modest,
- output length is non-trivial,
- draft model is much cheaper than target model,
- draft/target predictions are well aligned,
- acceptance rate is high,
- verification can be batched efficiently.

It may help less or hurt when:

- concurrency is already high and GPU is saturated,
- draft model is too expensive,
- acceptance rate is low,
- sequence lengths or sampling settings make drafts unreliable,
- extra KV/cache/memory overhead reduces serving capacity.

## EAGLE

EAGLE stands for Extrapolation Algorithm for Greater Language-model Efficiency.

The main beginner idea:

```text
instead of using a completely separate small LM as drafter,
EAGLE trains a lightweight draft module using target-model internal features
```

EAGLE predicts future information based on hidden states/features, then generates draft tokens that the target model verifies.

Why this can work:

- target-model internal features contain strong information about likely next tokens,
- a lightweight EAGLE drafter can be cheaper than the target model,
- better alignment can improve acceptance rate.

## EAGLE-2

EAGLE-2 adds context-aware dynamic draft trees.

Earlier speculative methods often used a static draft tree:

```text
same draft tree shape regardless of context
```

EAGLE-2 observes that acceptance rates are context-dependent.

Beginner version:

```text
some contexts have confident drafts
some contexts are uncertain
so the draft tree should adapt to the context
```

EAGLE-2 uses draft-model confidence to build a dynamic tree, spending more speculation budget where tokens are more likely to be accepted.

## EAGLE-3

EAGLE-3 further changes the drafting approach.

The paper summary describes two high-level changes:

- direct token prediction instead of feature prediction,
- multi-layer feature fusion rather than relying only on top-layer features.

Beginner interpretation:

```text
EAGLE-3 tries to make the drafter more accurate and scalable,
so more draft tokens can be accepted by the target model.
```

For framework learning, the important point is not to memorize every EAGLE variant. The important point is:

```text
EAGLE-style methods improve speculative decoding by building a better drafter,
which improves acceptance rate and therefore decode speed.
```

## MTP: Multi-Token Prediction

MTP means Multi-Token Prediction.

Instead of only predicting the next token, a model is trained or augmented to predict several future tokens.

Beginner mental model:

```text
normal LM head:
predict token t+1

MTP:
also predict token t+2, t+3, ...
```

DeepSeek-V3 uses a multi-token prediction objective during training and notes that it can also support speculative decoding for inference acceleration.

In serving, MTP can act like a built-in drafter:

```text
MTP heads/modules propose multiple future tokens
target model verifies or accepts them according to the serving algorithm
```

Compared with an external small draft model, MTP may reduce deployment complexity because the future-token prediction capability is built into or attached to the model family.

## EAGLE vs MTP

Useful beginner comparison:

```text
EAGLE:
train/use a draft module based on target-model features

MTP:
model has heads/modules/objective for predicting multiple future tokens
```

Both are ways to generate draft tokens for speculative decoding.

Both still need careful verification/acceptance logic if the goal is to preserve target-model output behavior.

## Framework Support

vLLM documentation lists speculative decoding methods such as:

- EAGLE,
- MTP,
- draft models,
- PARD,
- MLP,
- n-gram and suffix decoding.

SGLang documentation lists:

- EAGLE-2/EAGLE-3,
- MTP,
- classic draft-model decoding,
- n-gram variant,
- DFLASH,
- other speculative decoding options.

The exact command line and model compatibility can change quickly, so always check framework docs before running.

## Metrics To Track

Speculative decoding should not be evaluated only by average tokens/s.

Track:

- acceptance rate,
- TTFT,
- TPOT,
- inter-token latency,
- p95/p99 latency,
- output tokens/s,
- GPU memory overhead,
- draft model overhead,
- target verification time,
- effect under different concurrency.

The most important practical question:

```text
Does speculation reduce latency for my workload after accounting for drafting overhead?
```

## Interview Sentence

```text
Speculative decoding 的核心是 draft-then-verify。一个更便宜的 drafter 先猜多个未来 token，然后大模型 target 在一次并行 forward 里验证这些 token。如果 draft tokens 被接受，就相当于一次 target forward 生成了多个 token，从而降低 decode latency。关键指标是 acceptance rate：接受率越高，每次昂贵 target verification 能提交的 token 越多，收益越大。

EAGLE 和 MTP 都是在改善 drafter。EAGLE 系列通过利用 target model 的内部 feature 或更强的 draft module 来提高 draft token 质量，EAGLE-2 引入 context-aware dynamic draft tree，EAGLE-3 进一步改进 token prediction 和 feature fusion。MTP 则是让模型具备 multi-token prediction 能力，可以预测多个未来 token，用作 speculative decoding 的 draft 来源。

但 speculative decoding 不是所有场景都一定更快。它有 draft overhead、额外显存和调度复杂度。如果并发已经很高、GPU 已经饱和，或者 acceptance rate 很低，收益可能变小甚至变差。所以我会重点看 acceptance rate、TPOT、p95/p99 latency、GPU memory overhead 和不同 concurrency 下的表现。
```

## Sources

- Speculative sampling paper: https://arxiv.org/abs/2302.01318
- EAGLE paper: https://arxiv.org/abs/2401.15077
- EAGLE-2 paper: https://arxiv.org/abs/2406.16858
- EAGLE-3 paper: https://arxiv.org/abs/2503.01840
- DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437
- vLLM speculative decoding docs: https://docs.vllm.ai/en/stable/features/speculative_decoding/
- SGLang speculative decoding docs: https://docs.sglang.ai/advanced_features/speculative_decoding.html
