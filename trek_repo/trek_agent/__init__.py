"""
trek_agent — a clean, Bedrock-native function-calling travel-planning agent for the TREK benchmark.

Replaces the monolithic scratchpad-ReAct engine (llm_pipeline.py), which the 2026-07-20 readiness
audit found structurally unfit for the full run (drops scorer-required fields, no real notebook, no
final-planning step, /travel_time 404, stale queries, leaked keys). Design goals:

  * ONE provider path: AWS Bedrock Converse API with native tool use (toolConfig) — uniform across
    every Bedrock-hosted model (Claude, Nova, Llama, Mistral, DeepSeek, ...).
  * LOSSLESS working memory: full tool results are kept in conversation history verbatim (every
    scorer-required field: flight times, open_hours, duration, lat/lon, check-in), so the model never
    has to fabricate a time it was never shown. Plus a model-writable notebook (write_note).
  * Explicit final-planning turn before submit_plan.
  * SANDBOX framing: the KB is the ground truth; never refuse as "fake data" — only the 3 typed
    impossibility classes are valid refusals.
  * compute_travel_time wired as a free, non-billable tool (the exact model B3 scores with).
  * Per-model max_tokens (generous; reasoning models get room to think + emit the full plan).
  * Credentials loaded at runtime, never printed or serialized.
"""
