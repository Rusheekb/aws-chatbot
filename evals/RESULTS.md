# Evaluation results

How the Learn2Lead assistant went from **88% to 98%** answer accuracy, measured on 32 test questions. Every number here comes from `python3 -m evals.run_eval`; the raw results (every answer and the judge's reasoning) are in [`results/`](results/).

## How it's measured

- **32 questions** in [`questions.json`](questions.json): new visitors (12), students and parents (10), tutors (4), off-topic requests (3), questions the documents can't answer (2), and an account question the bot must not pretend to answer (1).
- **Retrieval** (27 questions that have an answer in the documents):
  - *Hit rate*: an expected document is in the search results.
  - *MRR* (mean reciprocal rank): 1.0 if it's always first, 0.5 if always second.
  - *Evidence recall*: a phrase that actually answers the question is in the retrieved text. This is stricter than hit rate: finding the right file but the wrong chunk doesn't count. All evidence phrases were checked against the documents by script.
- **Answers**: each answer is graded against a reference answer by a separate model (Nova Pro, temperature 0). For off-topic questions it must decline; for unanswerable ones it must say so without inventing anything. Any answer containing `<thinking>` tags fails.
- **Every failure was also read by a person.** The judge isn't perfect: in one case it called the real contact email "invented".

## Summary

| | Baseline | Final |
|---|---|---|
| Answer accuracy | 88% (28/32, both runs) | **98%** (32/32 and 31/32) |
| Retrieval hit rate | 96% | **100%** |
| Evidence recall | 85% | **96%** |
| Cost per question | $0.00099 | $0.00119 (+20%) |
| Input tokens per question | 2,472 | 2,996 |

The accuracy gain came with a cost: returning 5 larger chunks instead of 3 small ones sends about 500 more input tokens per question. At about $0.0012 per question, 1,000 visitor questions cost about $1.20.

## Retrieval experiments

Retrieval experiments only call the embedding model, so they cost almost nothing and were run first. "Top" is how many chunks the search returns.

| Change | Chunk words | Top | Search | Hit rate | MRR | Evidence recall |
|---|---|---|---|---|---|---|
| **Baseline** | 60 | 3 | semantic | 96% | 0.94 | 85% |
| Smaller chunks | 40 | 3 | semantic | 96% | 0.92 | 74% |
| Larger chunks | 100 | 3 | semantic | 93% | 0.84 | **93%** |
| | 150 | 3 | semantic | 93% | 0.83 | 89% |
| | 250 | 3 | semantic | 85% | 0.74 | 81% |
| Contextual chunk headers | 60 | 3 | semantic | 85% | 0.83 | 81% ❌ |
| | 100 | 3 | semantic | 81% | 0.81 | 81% ❌ |
| + Help Center search keywords | 60 | 3 | semantic | 96% | 0.92 | 85% |
| | 100 | 3 | semantic | 93% | 0.83 | 93% |
| + Hybrid search (BM25 + embeddings) | 60 | 3 | hybrid | 93% | 0.91 | 78% |
| | 100 | 3 | hybrid | 96% | 0.82 | 93% |
| + Top 5 | 100 | 5 | semantic | 93% | 0.83 | 93% |
| | 100 | 5 | hybrid | **100%** | 0.83 | **100%** |
| Section-aware chunking (best variant) | 100 | 5 | semantic | 96% | 0.92 | 89% ❌ |
| **Final** (prices also written as a sentence) | 100 | 5 | hybrid | **100%** | 0.84 | **96%** |

What I learned:

1. **Chunk size is a trade-off.** Small chunks rank the right document first (high MRR); medium chunks are more likely to contain the whole answer (high evidence recall). On these documents 100 words was best, while on the garden test documents in lesson 15 smaller was better. The right size depends on the documents.
2. **Hit rate alone was misleading.** It said retrieval was 96% good while evidence recall showed the answer was missing from 15% of results.
3. **Contextual chunk headers made things worse.** Prefixing chunks with titles like "Apply to tutor" let the word "tutor" dominate short chunks, so "How much does tutoring cost?" matched the tutor application guide. A popular technique isn't automatically right for your data.
4. **Hybrid search only helped together with keywords and top 5.** The ablation at 100 words and top 5: semantic 93% vs hybrid 100% evidence recall. Keyword search is what found "become a tutor" (from the Help Center's own search keywords) and "30-minute class".
5. **Section-aware chunking (a new chunk at every heading) didn't help.** It created many tiny chunks without context.
6. **Tables embed poorly.** The prices lived only in a Markdown table, and editing the text above it moved the table into a chunk that never ranked first. Writing the prices as one sentence fixed it in every configuration.
7. **Editing content can break retrieval elsewhere.** Clarifying the savings line fixed one question and broke the price question, because chunk boundaries moved.

## Answer experiments

| Run | Change | Accuracy | Notes |
|---|---|---|---|
| baseline (×2) | 60 words, top 3, semantic | 88%, 88% | Same 4 failures both runs: prices, savings, child password, becoming a tutor |
| hybrid-100-top5 | Best retrieval + keywords | 94% | Prices, child password and tutor application fixed |
| fixes-run1 | Savings clarified in pricing doc; "answer the most likely meaning first" | 91% | Savings fixed, but the edit broke the price question (see 7 above) |
| best-retrieval (×2) | Prices also written as a sentence | 100%, 97% | One run solved a homework problem it should have declined |
| homework-fix (×2) | Stronger off-topic rule in the prompt | 97%, 94% | One run speculated about tutor background checks |
| **final (×2)** | "Never infer what the documents don't say" | **100%, 97%** | Only failure: an ambiguous question (below) |

### Targeted tests for rare failures

Some failures only happen some of the time, so a single run can't show whether a fix works:

- **Solving homework instead of declining:** 5/6 correct with the original prompt → **10/10** after naming the exact behaviors to avoid ("never solve problems, show steps or write essays") and giving the reason.
- **Speculating about background checks:** 5/5 before and 5/5 after the "never infer" rule. The failure is too rare to measure with this many runs, so this fix is **unproven**. It was kept because it states the intended behavior and caused no regressions.

### Known limitation

"I'm a parent. How do I get my kid into their class?" failed in 5 of the 8 Nova 2 Lite runs since the retrieval change (it passed both baseline runs). The bot reads "get into their class" as *enroll* (add a student, find a tutor) instead of *join today's class*. The saved results show where this is decided: in every failing run, the model's own search query brought back "Add a child" first; in every passing run, "Join your class" came first. So the interpretation happens when the model writes its search query, before it answers.

The "answer the most likely meaning first" rule didn't change this. I left the question as written, because real users ask ambiguous questions too. A possible fix is to have the bot ask a short clarifying question when two guides match equally well.

## Model comparison

Same 32 questions, same retrieval and prompt.

| Model | Accuracy | Content accuracy (tags removed) | Cost per question | Avg time |
|---|---|---|---|---|
| **Nova 2 Lite** (chosen) | **98%** | 98% | $0.0012 | 4.6s |
| Nova Micro | 0% | 91% | **$0.0001** | 1.8s |
| Nova Pro | 0% | 94% | $0.0020 | 4.0s |
| Llama 4 Scout | — | — | — | — |

- **Nova Micro and Nova Pro put `<thinking>` tags in every answer** when tools are available, so none of their answers could be shown to users as is.
- **Llama 4 Scout can't be used with this design**: Bedrock returns "This model doesn't support tool use in streaming mode."
- **Possible next step:** Nova Micro's content is 91% correct at about 1/12 of the cost. Removing `<thinking>` blocks from the stream (tricky, because a tag can be split across chunks) could make it the cheapest option. The eval would show whether that 7-point accuracy drop is worth the savings.

## Infrastructure finding

During experiments, a full run failed with `ThrottlingException` on the embedding model after hundreds of calls in a few minutes. Three fixes followed: adaptive retries on both Bedrock clients, a cache for query embeddings, and reporting AWS errors separately from accuracy so an outage can't look like a quality drop.

## Running it yourself

```bash
python3 -m evals.run_eval --retrieval-only                 # retrieval only, almost free
python3 -m evals.run_eval --label my-run                     # full run, about $0.05
SEARCH_MODE=semantic TOP_K=3 python3 -m evals.run_eval --retrieval-only    # try a setting
MODEL_ID=us.amazon.nova-micro-v1:0 INPUT_PRICE_PER_M=0.035 OUTPUT_PRICE_PER_M=0.14 \
    python3 -m evals.run_eval --label nova-micro             # try a model
```
