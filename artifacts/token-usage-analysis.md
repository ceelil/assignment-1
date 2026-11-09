# Token usage: compaction vs. no compaction

Task: `django__django-15368` (`bulk_update()` does not resolve plain `F(...)` expressions).
Model: `deepseek/deepseek-v4-flash-0731` through OpenRouter. All token counts are the
provider-reported `usage` values recorded in the two trajectories.

| Run | Trajectory | Compaction threshold |
|---|---|---|
| Compacted | `django__django-15368-trajectory.json` | 6,000 estimated tokens, keep last 2 actions |
| Baseline | `django__django-15368-baseline-trajectory.json` | off (`COMPACT_THRESHOLD=0`) |

Both runs fixed the bug the same way: in `QuerySet.bulk_update()` the check
`isinstance(attr, Expression)` became `hasattr(attr, "resolve_expression")`, so `F()`
(which subclasses `Combinable`, not `Expression`) is treated as an expression. The baseline
patch also removed the now-unused `Expression` import.
The compacted run's patch passes `make check-swebench INSTANCE=django__django-15368`
(`RESOLVED: yes`).

## Results

| Metric | Compacted | Baseline | Change |
|---|---:|---:|---:|
| ReAct steps (model calls) | 25 | 28 | −3 |
| Tool calls (`execute` / `invoke_skill` / `send_message`) | 29 / 1 / 1 | 31 / 1 / 1 | |
| Compactions | 3 (after steps 6, 10, 20) | 0 | |
| Prompt tokens, all action requests | 103,047 | 231,331 | −55% |
| Mean prompt per step | 4,122 | 8,262 | −50% |
| Largest single prompt | 5,880 | 13,361 | −56% |
| Prompt at the final step | 4,823 | 13,361 | −64% |
| Completion tokens (action requests) | 6,474 | 6,252 | +4% |
| Compaction requests: prompt + completion | 10,993 + 3,009 = 14,002 | 0 | |
| Total tokens incl. compaction | 123,523 | 237,583 | −48% |
| Prompt tokens served from cache | 79,872 (78%) | 211,968 (92%) | |
| Uncached prompt tokens | 23,175 | 19,363 | +20% |
| Reported cost, incl. compaction | $0.00705 | $0.00884 | −20% |

Prompt size per step (provider-reported prompt tokens):

```
step      1     5    10    15    20    25    28
compacted 1790  4449  5606  4514  5880  4823
baseline  1790  4881  7026  8560 10306 12244 13361
```

The baseline grows almost linearly: every step re-sends the whole history plus the new
action and observation (about 430 tokens per step on average). The compacted run is a
sawtooth. Each time the active context passed 6,000 estimated tokens, the old steps were
replaced by a model-written working memory and the prompt dropped to about 3,200 tokens
(estimated 6,578 → 2,279, 7,252 → 3,050, 7,165 → 2,301).

## Why the trends look this way

**Total prompt tokens fall much more than the number of steps.** Without compaction the
cost of step *n* is proportional to *n*, so the total over a run grows roughly with the
square of the run length. Compaction caps the size of each request, so the total grows
roughly linearly. The difference is already 2.2× on a 25-step task and would widen on
longer tasks. The run being 3 steps shorter accounts for little of it: the baseline's first
25 steps alone used 192,806 prompt tokens.

**Compaction has its own cost.** Each compaction is an extra model call that reads the old
steps (about 3,300–4,100 prompt tokens) and writes a summary of 700–1,200 tokens. Across
the run that was 14,002 tokens, or 11% of the compacted run's total. It still pays off here
because each compaction removes 3,000–4,000 tokens from every later request.

**Prompt caching narrows the cost gap.** The baseline's prompt is append-only, so each
request repeats the previous one as a prefix, and the provider served 92% of its prompt
tokens from cache. Compaction rewrites the beginning of the history, which invalidates the
cache: right after each compaction the cached share dropped to 1,280–1,792 tokens (only the
system and task messages). As a result the compacted run actually sent *more* uncached
prompt tokens (23,175 vs 19,363), and its dollar cost was only 20% lower even though it used
55% fewer prompt tokens. With a provider that does not cache, or with a longer task, the
saving would be larger.

**Completion tokens barely changed.** Compaction shrinks what the model reads, not what it
writes. The compacted run reasoned slightly more (3,339 vs 2,453 reasoning tokens), which
fits a model that occasionally has to re-orient from a summary rather than read the raw
history.

## Trade-offs

| | Compaction | No compaction |
|---|---|---|
| Context size | Bounded near the threshold | Grows every step, up to the model's limit |
| Total tokens | Grows about linearly with steps | Grows about quadratically with steps |
| Extra calls | One summarization call per compaction | None |
| Cache reuse | Lost after each compaction | Very high (append-only prefix) |
| Information | Old details survive only if the summary keeps them | Everything stays verbatim |
| Failure modes | A bad summary can drop a key fact (a path, an error message, a failed attempt) and cause repeated work | Long contexts dilute attention; a very long run eventually overflows the window |

In this run compaction did not hurt the outcome: the agent kept its plan across all three
compactions and reached the same fix in fewer steps. The working memories it relied on
recorded the culprit (`django/db/models/query.py`, the `isinstance(attr, Expression)` check
in `bulk_update`), why `F` fails it, and what had already been tried, which is the kind of
information the compaction prompt asks to keep. The risk shows up on tasks where an exact
detail from early in the run matters much later; keeping the most recent actions verbatim
and asking the summarizer to quote exact error strings and paths reduces, but does not
remove, that risk.
