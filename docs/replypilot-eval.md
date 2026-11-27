# ReplyPilot retrieval evaluation (SP-123)

How well does knowledge-base search find the right help article for real,
messy customer questions? This page records the method, the before/after
numbers for grouping results by article, the misses, and what to try next.

## Method

| Item | Value |
|---|---|
| Golden set | 40 questions, 6 categories (5 to 7 each), 5 styles (8 each: paraphrase, typo, Hinglish, vague, short). Split in SP-124 into `data/eval/retrieval_dev.jsonl` (25) and held-out `data/eval/retrieval_test.jsonl` (15) |
| Knowledge base | 18 articles in `data/help_center/` |
| Model | `sentence-transformers/all-MiniLM-L6-v2` (same for before and after) |
| Store | pgvector, cosine distance |
| Script | `python -m supportpilot.rag.evaluate` |

Questions were written in customer language, not copied from the articles.
Ambiguous questions list every fair answer (for example "where is my stuff"
accepts `kb-track-order` or `kb-late-delivery`).

Metrics, per question then averaged:

- **recall@1**: the first result is a correct article.
- **recall@3**: a correct article is anywhere in the top 3.
- **MRR**: 1 / rank of the first correct article, 0 if not found in the top 10.

## Change under test: group by article

Before, `store.search` returned the top k **chunks**. One long article split
into several chunks could fill all three slots and push a second relevant
article out. Now `store.search(..., group_by_article=True)` (the default)
fetches `k × 5` chunks, keeps the best-scoring chunk per article and returns
the top k **articles**.

## Results

| Metric | Before (no grouping) | After (grouped) | Change |
|---|---|---|---|
| recall@1 | 0.675 | 0.675 | +0.000 |
| recall@3 | 0.850 | 0.850 | +0.000 |
| MRR | 0.790 | 0.790 | +0.000 |

| Category | recall@3 before | recall@3 after | Change |
|---|---|---|---|
| account | 1.000 | 1.000 | +0.000 |
| billing | 1.000 | 1.000 | +0.000 |
| cancellation | 0.714 | 0.714 | +0.000 |
| delivery | 0.857 | 0.857 | +0.000 |
| other | 0.833 | 0.833 | +0.000 |
| refund | 0.714 | 0.714 | +0.000 |

| Style | recall@3 before | recall@3 after | Change |
|---|---|---|---|
| hinglish | 0.750 | 0.750 | +0.000 |
| paraphrase | 1.000 | 1.000 | +0.000 |
| short | 1.000 | 1.000 | +0.000 |
| typo | 0.750 | 0.750 | +0.000 |
| vague | 0.750 | 0.750 | +0.000 |

Misses after grouping (6):

| Question | Style | Expected | Got (top 3) | Reason |
|---|---|---|---|---|
| ghar ka pata galat daal diya, badal sakte ho | hinglish | kb-change-address | kb-business-hours, kb-update-payment, kb-invoice-download | Hinglish: English-only model has no meaning for Hindi words |
| paisa wapas kab aayega mere account mein | hinglish | kb-refund-timeline | kb-privacy-policy, kb-delete-account, kb-contact-support | Hinglish: English-only model has no meaning for Hindi words |
| i dont like what i got | vague | kb-refund-request | kb-privacy-policy, kb-partial-refund, kb-delete-account | Vague: too little signal to pick one article |
| cancle my membrship plz | typo | kb-cancel-subscription | kb-delete-account, kb-contact-support, kb-privacy-policy | Vocabulary gap: customer words (or typos) not in the article |
| dont want it anymore | vague | kb-cancel-order, kb-refund-request | kb-privacy-policy, kb-delete-account, kb-cancel-subscription | Vague: too little signal to pick one article |
| custmer care numbr | typo | kb-contact-support, kb-business-hours | kb-late-delivery, kb-cancel-subscription, kb-cancel-order | Vocabulary gap: customer words (or typos) not in the article |

## What to improve next

1. **Add customer synonyms to articles.** A short "Also asked as" line per
   article (e.g. GST bill, money back, paisa wapas, membership) closes the
   vocabulary and Hinglish gaps cheaply, with no model change.
2. **Try a multilingual embedding model** such as
   `paraphrase-multilingual-MiniLM-L12-v2` (also 384-dim, so no schema change).
   Re-run this eval and re-baseline the gate if it wins.
3. **Hybrid search**: blend BM25 keyword scores with vector scores so exact
   terms like "invoice" or "GST" are not lost, and fall back to
   `kb-contact-support` when the top score is low.

## Dev / held-out split (SP-124)

The 40 questions are split so tuning cannot leak into the gate:

| Set | File | Questions | Use |
|---|---|---|---|
| Dev | `retrieval_dev.jsonl` | 25 | Tuning. Default for `python -m supportpilot.rag.evaluate`. Look at these misses freely. |
| Held-out | `retrieval_test.jsonl` | 15 | CI gate only. Never inspect its misses while tuning. |

The held-out set has 3 questions per style and at least 2 per category, and
keeps the same miss rate as the full set (2 of 15 vs 6 of 40).

Held-out baseline (`retrieval_test_baseline.json`, grouped retrieval,
all-MiniLM-L6-v2): recall@1 0.667, recall@3 0.867, MRR 0.791.

## Regression gate

`tests/test_rag_eval.py::test_heldout_retrieval_does_not_regress` runs in the
`test-postgres` CI job with the real model. It ingests the help centre, scores
the **held-out set only** and fails if recall@3 or MRR drops more than 0.05
below `retrieval_test_baseline.json`. With 15 questions one extra miss costs
0.067 recall@3, so in practice the gate allows no new held-out misses. The job
sets `REQUIRE_RETRIEVAL_GATE=1`, so the gate can never be silently skipped.

To change the model or the question sets intentionally, regenerate the
baseline in the same PR:

```bash
python -m supportpilot.rag.evaluate --golden data/eval/retrieval_test.jsonl \
  --json data/eval/retrieval_test_baseline.json
```
