# Decision as Reranker
Typed decision model as Reranker for RAG

## What is it?

### Recall the typed decision model

From [TypeSafe](https://docs.typesafe.ai/concepts/system-one), a decision model like Jev understands natural-language input. It returns typed decisions and probabilities rather than generated text.

Possible answers are defined through primitives. A decision model never emits a value outside the option set you supplied, so there is nothing to parse and nothing to hallucinate into a schema.

| Primitive | Question it answers | What you define (`criteria`) | What comes back | Use it when |
| --- | --- | --- | --- | --- |
| `choice` | Which of these options? | A map `option -> description`; options are unordered. Jev accepts up to 255. | `choice` (argmax option), `probabilities` (one per option, sums to 1), `confidence` | The answer space is a fixed, unordered label set: routing, intent, document type, tool selection. |
| `score` | Which level, on an ordered spectrum? | An ordered list of level descriptions, low to high (2-10 for Jev). Each level must be described in words, not numbered. | `score` (position on the level line, may fall between levels), `probabilities` per level, `legend`, `confidence` | The answer is a position you can describe in steps: severity, urgency, relevance. |
| `noul` | Is this true? | `instructions` holding one yes/no question or statement; optional `true` / `false` descriptions that fix the boundary. | `noul` = P(yes), a single float in `[0, 1]`. No separate `confidence`: with two outcomes the probability already describes the answer completely. | The probability itself is the signal: gating, filtering, abstention checks — and ranking, where you sort on the value instead of thresholding it. |

`score` is the probability-weighted mean of the level indices, $\text{score} = \sum_i i \cdot p_i$, so the same score can come from different distributions (all mass on level 1, or half on 0 and half on 2). Read `probabilities` next to it.

A fourth primitive, `Box` (localize something in an image), exists in the same family; it is out of scope here because Jev takes text input only.

### Recall reranker model

Reranker model receives a pair of inputs: query and retrieved information, score them (typically 0-1), then ranks information based on this relevance score.

## Workflow

### 1. Prepare a decision model

We turn an existing LLM into a decision model

#### 1.1 Training-free
[AnyJev](https://arxiv.org/abs/2610.00831) by Zhang et al. introduce a training-free way to turn an LLM into a decision model

#### 1.2 Lora adapter
[Kev](https://github.com/jaredpalmer/kev) is a family of decision models built on Qqwen3.5 and Qwen3.8

### 2. Serving
We standardize the interface of interacting with decision model to TypeSafe-compatible API.
You can use typesafe's API or host it yourself

### 3. Adapting decision output to reranking task

#### 3.1 Noul as a scorer

Retrieval (BM25 or dense) narrows a corpus to a shortlist; it ranks by shared words or cosine, not by whether a passage actually answers the query. A `noul` question per (query, candidate) pair returns a comparable probability for every pair, and sorting on it is the re-rank step. TypeSafe's [re-ranking cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe) reports this on CLERC legal retrieval (3,565 passages, 40 queries, BM25 top-30 shortlist): top-1 5% -> 18%, top-10 38% -> 62%.

#### 3.2 Choice question as a reranker (implement in this repository)
We can view the reranking task a **choice question type**. In [noul approach](#21-noul-as-a-scorer), we calculate score for each chunk separately. In this approach, we can calculate all scores at once. We can define a choice question with all chunks as options, and the model will return the best chunk along with probabilities for each chunk. This allows us to efficiently rerank all candidates in a single pass.

With *noul approach*, each chunk is scored independenly. With choice primitive approach, we need to be aware of defects caused by placing all options in sequence, as mentioned in AnyJev.

##### 3.2.1 Label prior
Temporarily use solution from AnyJev to renormalised

$𝑝^{dbs} ∝ \tilde 𝑝 / \hat q^𝛼$

𝛼 = 0.75 respect default value from orignal work and adjustable as parameter in code

#### 3.2.2 Position bias
AnyJev rotate the option sequence and apply van der Corput sequence. The order of the options is partially maintained.

Instead, we can simply
- Shuffle the order of the option sequence (one permutation)
- Tune the number of permutation, k,  to control computation budget

> [!NOTE]
> Number of permutation (k) is not limited to K like rotation

