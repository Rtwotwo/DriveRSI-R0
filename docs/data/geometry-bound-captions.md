# Geometry-bound scene captions

Vehicle identities, positions, headings and annotation displacement are compiled from source whitebox frames. VLM observes only static appearance/environment in supplied photos. The v3.1 contract rejects extra keys and records scope-screen warnings. A keyword screen is not proof of factual accuracy.

## Information scopes
retrieval_history: 21 whitebox samples, -2..0 s; photo inputs -1.5..0 s at source sampling times. No future image/track in this output.
video_clip: 71 whitebox samples, -2..+5 s; four sampled source times across four cameras. Intermediate whitebox frames are offline interpolation, not 10Hz measured ground truth. Missing annotation is not actor exit. Sample interval endpoints are represented bounds, not bounds of direct observation.

The fixed anchor ego frame is x forward, y left, z up. Class colors are semantic labels, never paint or subtype. Full tracks and calibrated cameras are in each scoped geometry JSON. This caption pilot does not load lane polygons or full scene occlusion geometry; complete_video_condition=false. A video consumer must add a verified map and visibility conditions. No generated video quality result is implied.

## Review and consumption
appearance_observations and candidate_claims preserve evidence-linked drafts. Quarantined claims are not in candidate prompts. draft_prompts may contain unreviewed appearance and are previews only. released_prompts deliberately contain no VLM appearance pending explicit evidence review. They are not visual training admission. Do not call format_valid factual_correct.

## Running
scripts/data/run_geometry_caption.py selects the cross-city pilot, prepares JPEG evidence and scoped jobs, and runs/resumes the shared private client. Use --first-scenes 4 to create an initial 8-call checkpoint; a new process without that option reuses those results. All credentials AND provider connection config must reside mode0600 within /path/to/private, directory0700. Neither public-network fallback nor environment-key fallback is accepted. max_requests is a lifetime attempt cap per batch directory. Uncertain calls require review and are not blindly resent. HTTP429 alone has bounded retries. Rejected responses retain billed usage.

scripts/data/analyze_geometry_captions.py measures pilot coverage, token cost estimates, retrieval probes and draft diversity. The structural probe uses its own predicates for labels and is a wiring test. The separate appearance probe has only16 assistant-reviewed anchor mosaics. Neither is a general benchmark.

## Full batch preparation
exp/caption-geometry64-v1/full-batch-plan has103288 officialnavtrain members in1614 planned shards (64 each) and206576 nominal scope calls. It is a plan, not an active dispatcher. Source/camera preflight, explicit private-service budgets, shard execution and appearance release review remain mandatory before dispatch. No navtest inputs are accessed. No API/model configuration is copied into this document or the worktree.
