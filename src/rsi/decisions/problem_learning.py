"""Contracts for one failed query, multiple sources, and new-only native SFT."""
from collections import Counter
import math


def select_problem_sources(candidates, round_index, policy):
    """Similarity-gap budget, not calibrated semantic accuracy or a fixed count."""
    if not candidates or not 0 <= round_index < policy['max_rounds']:
        raise ValueError('Bounded nonempty problem source pool required')
    ranked=sorted(candidates,key=lambda r:(-r['max_similarity'],r['token']))
    best=ranked[0]['max_similarity']
    margin=policy['similarity_margins'][round_index]
    selected=[r for r in ranked if best-r['max_similarity'] <= margin]
    return selected[:policy['source_caps'][round_index]]


def source_quota(required_tokens, origins, target):
    """Count admitted, deduplicated samples by source, never by global total."""
    if (not required_tokens or len(set(required_tokens)) != len(required_tokens)
            or type(target) is not int or target < 10):
        raise ValueError('Unique selected sources and at least10 samples per source required')
    counts = Counter(origins)
    extra = sorted(set(counts) - set(required_tokens))
    missing = {t: max(0, target-counts[t]) for t in required_tokens if counts[t] < target}
    return dict(passed=not extra and not missing, counts=dict(counts),
                required_per_source=target, required_sources=len(required_tokens),
                required_total=len(required_tokens)*target, missing=missing, unexpected_sources=extra)


def sft_updates(sample_count, epochs, global_batch):
    if any(type(x) is not int or x < 1 for x in (sample_count, epochs, global_batch)):
        raise ValueError('Positive dataset, epochs and batch size required')
    updates = math.ceil(sample_count/global_batch)*epochs
    if updates % 2 or updates > 256:
        raise ValueError('Two recovery phases require even updates within256')
    return updates


def target_success(score, threshold=.95, *, strict=False):
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Finite normalized target PDMS required')
    return score > threshold if strict else score >= threshold
