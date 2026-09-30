# Speculative decoding ownership

This package generates **next-round draft tokens**. It does not own target-model
verification or scheduler progress. The flow follows the vLLM V1 split:

1. After a sampled output is committed, `Scheduler.prepare_next_draft` calls a
   proposer and stores its tokens and optional draft probabilities on the request.
2. On the next round, `Scheduler.begin_speculative_batch` consumes that proposal,
   caps it to the token budget, and reserves target KV/GDN trial capacity.
3. `ModelRunner.verify_speculative_batch` runs the target model over the anchor
   and candidates. `sampling/rejection_sampler.py` selects accepted candidates
   and a recovery or bonus token.
4. `Scheduler.finish_speculative` commits accepted progress, trims rejected trial
   KV/GDN state, and prepares the following draft for unfinished requests.

| Location | Responsibility |
|---|---|
| `spec_decode/{config,interfaces,metadata,factory,metrics}.py` | Draft configuration, proposal contracts, backend selection, counters |
| `spec_decode/llm_base_proposer.py` | Shared model-backed feature history, draft KV, request-slot lifecycle and device proposal packing |
| `spec_decode/{ngram,mtp,eagle3,block}.py` | N-gram and model-specific draft proposers |
| `models/{qwen3_5_mtp,eagle3,block_draft}.py` | Draft model architectures and weight loading |
| `scheduler/scheduler.py` | Cached draft consumption, token budgeting, KV/GDN reservation and rejection rollback |
| `engine/model_runner.py`, `engine/spec_verification/` | Target forward pass, feature capture, verification metadata, trial state and commit |
| `sampling/{rejection_sampler,batch_verifier}.py` | Acceptance, rejection, recovery and bonus sampling |
| `engine/kv_cache_manager.py` | Target KV blocks and GDN snapshots |

Draft generation currently runs after CPU bookkeeping for every proposer. The
proposer interface and request-level cached proposal leave room to generate
device-only drafts earlier in the runner's sampling stage.

`MTPProposer`, `EagleProposer` and `BlockProposer` inherit directly from
`SpecDecodeBaseProposer`; block models override candidate generation while
EAGLE and block models enable target intermediate-feature capture. The
`Eagle3Proposer` name remains an alias for existing callers.
