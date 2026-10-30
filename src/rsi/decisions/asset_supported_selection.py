"""Choose available training assets only inside a bounded global visual result.

Asset availability never licenses extending the visual cutoff. Coordinates,
heading, map identity and future training targets are outside this interface.
"""
import numpy as np


def select(query, matrix, members, available_tokens, *, top_k=20, limit=3):
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("Raw visual cutoff must be from1 to20")
    if type(limit) is not int or not 1 <= limit <= top_k:
        raise ValueError("Bounded output limit required")
    q, x = np.asarray(query), np.asarray(matrix)
    if q.ndim != 1 or x.ndim != 2 or x.shape != (len(members), len(q)):
        raise ValueError("Aligned query, matrix and training membership required")
    tokens = [m["token"] for m in members]
    if len(set(tokens)) != len(tokens) or not set(available_tokens) <= set(tokens):
        raise ValueError("Unique training membership and known assets required")
    scores = x @ q
    if not np.isfinite(scores).all() or not np.isfinite(q).all():
        raise ValueError("Finite visual features required")
    if abs(np.linalg.norm(q)-1) > .001 or np.any(np.abs(np.linalg.norm(x,axis=1)-1)>.001):
        raise ValueError("The frozen cosine feature contract requires unit vectors")
    # Stable global ranking is established BEFORE any asset/source-group filter.
    ranking = sorted(range(len(tokens)), key=lambda i:(-float(scores[i]),tokens[i]))[:top_k]
    selected, groups = [], set()
    for rank, index in enumerate(ranking,1):
        row = members[index]
        if (row["token"] not in available_tokens or row["history_continuous"] is not True
                or row["source_group"] in groups):
            continue
        selected.append(dict(token=row["token"], source_log=row["source_log"],
            source_group=row["source_group"], raw_visual_rank=rank,
            visual_similarity=float(scores[index]), localization_used_for_selection=False))
        groups.add(row["source_group"])
        if len(selected)==limit:
            break
    return selected
