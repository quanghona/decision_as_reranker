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

## Getting started

Install the package with the extra matching your serving instance:

```bash
uv sync --extra typesafe   # or --extra openai
```

There are two ways to use the reranker: call it directly from Python, or run it as a vLLM/Cohere-compatible HTTP server.

### 1. Run in Python

Inject a backend (the transport) into a reranker (the questions, parsing and ordering). The `noul` primitive scores each candidate independently in a single request:

```python
from typesafe_sdk import TypeSafeClient
from decision_as_reranker.backend.typesafe import TypeSafeBackend
from decision_as_reranker.reranker import DecisionNoulReranker

backend = TypeSafeBackend(
    TypeSafeClient(api_key="local", base_url="http://localhost:8009", model="kev-latest"),
)
reranker = DecisionNoulReranker(
    backend,
    instructions="Could the candidate passage be what the query asks for?",
    noul_true="It states the specific fact or rule the query is asking about.",
    noul_false="It merely shares the query's topic.",
)
ranked = reranker.rerank(query, candidates)  # list[ScoredCandidate], best first
print(ranked[0].candidate, ranked[0].score, ranked[0].index)
```

The `choice` primitive ranks the whole shortlist with one joint comparison (**experimental**, see [3.2](#32-choice-question-as-a-reranker-implement-in-this-repository)) — `k` orderings averaged to damp position bias, `alpha` controlling the label-prior correction:

```python
from decision_as_reranker.reranker import DecisionChoiceReranker

reranker = DecisionChoiceReranker(
    backend,
    instructions="Which of these candidate passages is the one the query asks for?",
    k=4,
    alpha=0.75,
)
ranked = reranker.rerank(query, candidates)
```

To point at OpenAI's Decisions endpoint instead, swap the backend:

```python
from openai import OpenAI
from decision_as_reranker.backend.openai import OpenAIBackend

backend = OpenAIBackend(OpenAI(), model="gpt-6-luna")
```

### 2. Serve as a vLLM reranker

`serve_vllm.py` exposes the same ranking behind vLLM's rerank API (`POST /rerank`, `/v1/rerank`, `/v2/rerank`), so any RAG framework that speaks that dialect can use a decision model by changing only a URL:

```bash
# noul, against a local Kev
uv run serve_vllm.py --backend typesafe --endpoint http://localhost:8009 --model kev-latest

# the choice primitive, 4 orderings
uv run serve_vllm.py --primitive choice --k 4 --alpha 0.75

# OpenAI's Decisions endpoint
uv run serve_vllm.py --backend openai --model gpt-6-luna --api-key $OPENAI_API_KEY
```

Then, from any client:

```bash
curl -s localhost:8000/v1/rerank -H 'content-type: application/json' \
     -d '{"query": "what is the capital of France?", "documents": ["Paris is the capital of France.", "Berlin is German."], "top_n": 2}'
```

`--help` lists every knob; the ranking flags mirror the Python arguments above.

## Workflow

### 1. Prepare a decision model

We turn an existing LLM into a decision model. There is no single right one: the open field moves fast, [`multimodalart/jev-decision-index`](https://huggingface.co/spaces/multimodalart/jev-decision-index) is a community leaderboard that ranks every public Jev reproduction on a chance-corrected Decision Index (private tests, so the ranking cannot be gamed).

Three ways to get a decision model, in increasing effort:

#### 1.1 Training-free
[AnyJev](https://arxiv.org/abs/2610.00831) by Zhang et al. turn a stock LLM into a decision model at inference time, with parallel constrained decoding and no new weights. On the index this is the "Decoding" category; the published `morriszjm/AnyJev-Tacitus-31B` is the reference entry, and it is also the source of the label-prior and position-bias corrections the `choice` reranker below reuses.

#### 1.2 LoRA adapter
[Kev](https://github.com/jaredpalmer/kev) is a family of small decision models (LoRA adapters plus a pointer head on Qwen3.5 / Qwen3.8) that speak the TypeSafe System One API unchanged, so the `typesafe` backend in this repo points at a local Kev with no code change. Four sizes ship together as Kev 1.0:

Other promising entrants on the index, by kind: `Cloudflare/clef` and `Cloudflare/clef-flash` (small, fast), the `vllm-sr/d3-*` family, `PelaAI/KnowLine-*`, `fastino/GLiDE-*`, `h2oai/h2o-lightning-*`, and `perplexity-ai/pplx-decider-v1-27b`. Most expose a Jev-compatible endpoint; check each card for whether it speaks `/v1/systemone` (the `typesafe` backend) or an OpenAI-compatible API (the `openai` backend).

#### 1.3 Fine-tuning
When your questions look different from the public suites — your own domains, your own labels — a short fine-tune usually beats any prompt change. Kev ships a `--init_from` path that starts from a released checkpoint and keeps what it already knows; a coding-agent skill runs the whole loop on Modal. Size your dataset first: with a few hundred records the gain is inside the noise.

### 2. Serving
We standardize the interface of interacting with decision model to TypeSafe-compatible API.
You can use typesafe's API or host it yourself

### 3. Adapting decision output to reranking task

#### 3.1 Noul as a scorer

Retrieval (BM25 or dense) narrows a corpus to a shortlist; it ranks by shared words or cosine, not by whether a passage actually answers the query. A `noul` question per (query, candidate) pair returns a comparable probability for every pair, and sorting on it is the re-rank step. TypeSafe's [re-ranking cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe) reports this on CLERC legal retrieval (3,565 passages, 40 queries, BM25 top-30 shortlist): top-1 5% -> 18%, top-10 38% -> 62%.

#### 3.2 Choice question as a reranker (implement in this repository)

> [!WARNING]
> **Experimental.** The `choice` reranker is not validated on a retrieval benchmark the way the `noul` path is (the CLERC numbers above are `noul`). Joint comparison gives up the per-candidate independence that makes `noul` scores stable, and trades it for two biases that need correcting (below) and a context budget that grows with the shortlist. Treat it as a research option, not a drop-in default.

We can view the reranking task a **choice question type**. In [noul approach](#21-noul-as-a-scorer), we calculate score for each chunk separately. In this approach, we can calculate all scores at once. We can define a choice question with all chunks as options, and the model will return the best chunk along with probabilities for each chunk. This allows us to efficiently rerank all candidates in a single pass.

With *noul approach*, each chunk is scored independenly. With choice primitive approach, we need to be aware of defects caused by placing all options in sequence, as mentioned in AnyJev.

##### 3.2.1 Label prior
Temporarily use solution from AnyJev to renormalised

$𝑝^{dbs} ∝ \tilde 𝑝 / \hat q^𝛼$

𝛼 = 0.75 respect default value from original work and adjustable as parameter in code

#### 3.2.2 Position bias
AnyJev rotate the option sequence and apply van der Corput sequence. The order of the options is partially maintained.

Instead, we can simply
- Shuffle the order of the option sequence (one permutation)
- Tune the number of permutations, k, to control computation budget

> [!NOTE]
> Number of permutation (k) is not limited to K like rotation

##### 3.2.3 Cost: k and K

Two quantities, easy to confuse because both look like "how many":

- **K** — the number of options, i.e. the shortlist length. Fixed by the input, never a knob. It sets the *context* of every content request: each one carries the whole shortlist (state + K passage copies), so a long shortlist can overflow the model's window regardless of how few requests you make. This is why `serve_vllm.py` rejects an over-long shortlist with `--max-documents` rather than letting the backend refuse it mid-ranking.
- **k** — the number of orderings averaged per rerank. The tunable *compute* budget. `k = 1` is the plain joint question; `k = K` recovers AnyJev's full rotation. Effectively capped by `K!`, the number of distinct orderings that exist.

Requests per rerank, independent of K:

| configuration | requests | what each carries |
| --- | --- | --- |
| `noul` | 1 | K questions, one per candidate (independent) |
| `choice`, `alpha = 0` | k | one ordering of the K options each |
| `choice`, `alpha > 0` | 2k | k content requests (full passages) + k probe requests (option names only, no passage text) |

So `choice` costs `k` (or `2k`) round trips versus `noul`'s one — but each `choice` content request still carries all K passages, so context grows with K either way. The probe half of the `2k` is cheap on context (names only). Set `alpha = 0` to halve the request count, or `k = 1` to ask a single joint question; with one ordering the probe still removes that ordering's combined label-and-position bias, which is most of the correction for one extra request.
