"""Reserved tokens for abstraction bookkeeping."""

# 记入 abstracted_subsets：叙事相干门控判定不通过，占位避免同一频繁子集死循环挖矿。
SKIPPED_COHERENCE_FINGERPRINT_ID = "__skipped_narrative_coherence__"
# LLM 空响应 / JSON 解析失败等：占位避免长跑在同一子集上反复失败。
SKIPPED_SYNTHESIS_FAILURE_ID = "__skipped_synthesis_llm_failure__"
# 子集在日志上频繁但无法落地合成（缺证据 / DB 过滤后不足 min_size）：占位避免刷新环死循环。
SKIPPED_ABSTRACTION_MINING_NOOP_ID = "__skipped_abstraction_mining_noop__"
