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
