"""Training-only visual/semantic rank fusion; model scores never rank neighbors."""

import json
import math
import re
from collections import Counter, defaultdict

from .vectors import VectorStore

RRF_K = 60
STOP_WORDS = frozenset("a an the and or of on in at to with is are image cam f0 l0 r0 visible unknown this that scene shows there front side camera".split())


def fuse(lists):
    """Equal-weight reciprocal ranks; empty channels contribute no evidence."""
    values = defaultdict(float)
    for rows in lists:
        for rank, row in enumerate(rows, 1):
            values[row["token"]] += 1.0 / (RRF_K + rank)
    return [{"token": token, "fusion_score": value}
            for token, value in sorted(values.items(), key=lambda x: (-x[1], x[0]))]


class HybridRetriever:
    def __init__(self, catalog, vector_profile, caption_profile=None):
        if catalog.split != "navtrain":
            raise ValueError("Hybrid training retrieval requires navtrain")
        if not catalog.db.in_transaction:
            catalog.db.execute("BEGIN")
        self.catalog, self.vector_profile = catalog, vector_profile
        self.caption_profile = caption_profile
        self.vectors = VectorStore(catalog)
        self.vectors.profile_data(vector_profile)
        # A token-indexed join caused >6 minutes of random reads against the
        # production SQLite table, whose rows also contain large image vectors.
        # Scan metadata in insertion order once; do not rewrite any feature.
        ready = {row[0] for row in catalog.db.execute(
            "SELECT token FROM vectors NOT INDEXED WHERE profile=? AND status='ready'", (vector_profile,))}
        self.sources = {row["token"]: dict(row) for row in catalog.db.execute(
            "SELECT token,source_group,source_log FROM scenes") if row["token"] in ready}
        self.tags = {token: set() for token in self.sources}
        for row in catalog.db.execute("SELECT token,tag FROM tags"):
            if row["token"] in self.tags:
                self.tags[row["token"]].add(row["tag"])
        frequency = Counter(tag for tags in self.tags.values() for tag in tags)
        self.weights = {tag: math.log((1 + len(self.sources)) / (1 + count)) + 1
                        for tag, count in frequency.items()}
        self.labels = {}
        if caption_profile is not None:
            if catalog.db.execute("SELECT 1 FROM profiles WHERE id=?", (caption_profile,)).fetchone() is None:
                raise ValueError("Unknown caption profile")
            for row in catalog.db.execute(
                "SELECT a.id,a.token,a.observation FROM attempts a WHERE a.profile=? AND a.status='validated' "
                "AND a.id=(SELECT max(b.id) FROM attempts b WHERE b.token=a.token AND b.profile=a.profile) "
                "AND (SELECT verdict FROM reviews r WHERE r.attempt=a.id ORDER BY r.id DESC LIMIT 1)='spot_check_pass'",
                (caption_profile,)):
                if row["token"] in self.sources:
                    self.labels[row["token"]] = {"attempt_id": row["id"], "observation": json.loads(row["observation"])}

    def _diverse(self, rows, limit):
        result, used = [], set()
        for row in rows:
            group = self.sources[row["token"]]["source_group"]
            if group not in used:
                result.append(row)
                used.add(group)
                if len(result) == limit:
                    break
        return result

    def search_conditions(self, conditions, text="", *, limit=8, pool=64, exemplars=2):
        """Recall from training exposure conditions, without any student score.

        Visual exemplars are selected *inside navtrain*. Their appearance is a
        recall aid, not evidence that the same failure mechanism is present.
        """
        from driversi.rsi.decisions.scene_index import CONDITION_TAGS

        if (not isinstance(conditions, list) or not conditions
                or not set(conditions) <= CONDITION_TAGS
                or len(set(conditions)) != len(conditions)):
            raise ValueError("Known distinct history conditions required")
        if not isinstance(text, str) or len(text) > 2000:
            raise ValueError("Bounded caption query required")
        if not 1 <= limit <= pool <= 512 or not 1 <= exemplars <= 4:
            raise ValueError("Bounded retrieval limits required")
        query_tags = set(conditions)
        structured = []
        for token, tags in self.tags.items():
            shared = query_tags & tags
            if shared:
                structured.append({"token": token,
                    "condition_similarity": sum(self.weights[x] for x in shared) /
                        sum(self.weights.get(x, 1) for x in query_tags),
                    "shared_conditions": sorted(shared)})
        structured.sort(key=lambda x: (-x["condition_similarity"], x["token"]))
        structured = self._diverse(structured, pool)
        terms = sorted(set(re.findall(r"[a-z]{3,}", text.lower())) - STOP_WORDS)
        caption = []
        documents = {t: Counter(re.findall(r"[a-z]{3,}", x["observation"]["caption"].lower()))
                     for t, x in self.labels.items()}
        frequency = Counter(w for counts in documents.values() for w in counts)
        mean_length = sum(sum(x.values()) for x in documents.values()) / max(1, len(documents))
        for token, counts in documents.items():
            norm = 1.2 * (.25 + .75 * sum(counts.values()) / max(1, mean_length))
            value = sum(math.log(1 + (len(documents) - frequency[w] + .5) /
                                 (frequency[w] + .5)) * counts[w] * 2.2 / (counts[w] + norm)
                        for w in terms if counts[w])
            if value > 0:
                caption.append({"token": token, "bm25": value,
                                "attempt_id": self.labels[token]["attempt_id"]})
        caption.sort(key=lambda x: (-x["bm25"], x["token"]))
        caption = self._diverse(caption, pool)
        seeds = self._diverse(fuse([structured, caption]), exemplars)
        visual_lists = []
        for seed in seeds:
            visual_lists.append(self.vectors.search("training", seed["token"], self.vector_profile,
                limit=pool, shortlist=pool, exclude_source=True, diverse_sources=True)["results"])
        visual = self._diverse(fuse(visual_lists), pool)
        routes = {"visual": visual, "structured": structured, "caption": caption}
        ranks = {name: {r["token"]: i for i, r in enumerate(rows, 1)} for name, rows in routes.items()}
        selected = self._diverse(fuse(list(routes.values())), limit)
        for row in selected:
            token = row["token"]
            row.update(source_group=self.sources[token]["source_group"],
                shared_conditions=sorted(query_tags & self.tags[token]),
                ranks={k: v.get(token) for k, v in ranks.items()},
                caption_attempt=self.labels.get(token, {}).get("attempt_id"), trainable=False)
        return {"kind": "navtrain_condition_retrieval", "results": selected,
            "training_exemplars": [r["token"] for r in seeds],
            "conditions": conditions, "text": text,
            "routes": {k: v[:limit] for k, v in routes.items()},
            "ready_population": len(self.sources), "reviewed_caption_population": len(self.labels),
            "caption_channel_active": bool(caption), "vector_profile": self.vector_profile,
            "caption_profile": self.caption_profile, "scores_used_for_ranking": False,
            "test_samples_consumed": 0, "trainable": False,
            "claim": "exposure_condition_hypothesis_not_confirmed_failure_cause"}

    def search(self, query_token, *, limit=5, pool=128):
        if type(limit) is not int or type(pool) is not int or not 1 <= limit <= pool <= 512:
            raise ValueError("Require 1 <= limit <= pool <= 512")
        if query_token not in self.sources:
            raise ValueError("Query has no ready training feature")
        source = self.sources[query_token]["source_group"]
        eligible = {t for t, s in self.sources.items() if s["source_group"] != source}
        # Keep the existing visual model, exact cosine and view alignment unchanged.
        visual_report = self.vectors.search("training", query_token, self.vector_profile,
            limit=pool, shortlist=pool, exclude_source=True, diverse_sources=True)
        visual = visual_report["results"]
        query_tags = self.tags[query_token]
        structured = []
        for token in eligible:
            shared = query_tags & self.tags[token]
            if not shared:
                continue
            union = query_tags | self.tags[token]
            similarity = sum(self.weights[t] for t in shared) / sum(self.weights[t] for t in union)
            structured.append({"token": token, "condition_similarity": similarity,
                               "shared_conditions": sorted(shared)})
        structured.sort(key=lambda x: (-x["condition_similarity"], x["token"]))
        structured = self._diverse(structured, pool)
        caption = []
        query_label = self.labels.get(query_token)
        terms = []
        if query_label:
            terms = sorted(set(re.findall(r"[a-z]{3,}", query_label["observation"]["caption"].lower())) - STOP_WORDS)[:32]
            if terms:
                # All BM25 statistics come from reviewed captions, never rejected attempts.
                documents = {t: Counter(re.findall(r"[a-z]{3,}", x["observation"]["caption"].lower()))
                             for t, x in self.labels.items()}
                mean_length = sum(sum(x.values()) for x in documents.values()) / max(1, len(documents))
                frequency = Counter(word for counts in documents.values() for word in counts)
                for token in sorted(eligible & documents.keys()):
                    counts = documents[token]
                    score = 0.0
                    norm = 1.2 * (.25 + .75 * sum(counts.values()) / max(1, mean_length))
                    for term in terms:
                        count = counts[term]
                        if count:
                            idf = math.log(1 + (len(documents) - frequency[term] + .5) / (frequency[term] + .5))
                            score += idf * count * 2.2 / (count + norm)
                    if score > 0:
                        caption.append({"token": token, "bm25": score,
                                        "attempt_id": self.labels[token]["attempt_id"]})
                caption.sort(key=lambda x: (-x["bm25"], x["token"]))
                caption = self._diverse(caption, pool)
        semantic = self._diverse(fuse([structured, caption]), pool)
        all_fused = self._diverse(fuse([visual, semantic]), 2 * pool)
        fused = all_fused[:limit]
        # Preserve the measured baseline's candidates; semantic recall adds sources.
        retained = self._diverse([*visual[:limit], *semantic[:limit]], 2 * limit)
        retained_tokens = {row["token"] for row in retained}
        union = [row for row in all_fused if row["token"] in retained_tokens]
        # A different semantic representative of a visual source cannot evict the
        # selected visual representative during global source deduplication.
        present = {row["token"] for row in union}
        scores = {row["token"]: row["fusion_score"] for row in fuse([visual, semantic])}
        union.extend({"token": row["token"], "fusion_score": scores[row["token"]]}
                     for row in retained if row["token"] not in present)
        union.sort(key=lambda row: (-row["fusion_score"], row["token"]))
        ranks = {name: {x["token"]: i for i, x in enumerate(rows, 1)} for name, rows in (
            ("visual", visual), ("structured", structured), ("caption", caption), ("semantic", semantic))}
        for row in [*fused, *union]:
            token = row["token"]
            row.update(source_group=self.sources[token]["source_group"],
                       ranks={name: order.get(token) for name, order in ranks.items()},
                       shared_conditions=sorted(query_tags & self.tags[token]),
                       caption_attempt=self.labels.get(token, {}).get("attempt_id"), trainable=False)
        return {"kind": "navtrain_hybrid_retrieval", "query_token": query_token,
            "vector_profile": self.vector_profile, "caption_profile": self.caption_profile,
            "pool": pool, "limit": limit, "ready_population": len(self.sources),
            "eligible_population": len(eligible), "reviewed_caption_population": len(self.labels),
            "query_conditions": sorted(query_tags), "query_caption_attempt": query_label["attempt_id"] if query_label else None,
            "caption_query_terms": terms, "caption_channel_active": bool(caption),
            "semantic_sources": [name for name, rows in (("structured_conditions", structured), ("reviewed_caption_lexical", caption)) if rows],
            "fusion": {"algorithm": "reciprocal_rank_fusion", "constant": RRF_K,
                       "outer_weights": {"visual": 1, "semantic": 1}, "tie_break": "token"},
            "routes": {"visual": visual[:limit], "structured": structured[:limit],
                       "caption": caption[:limit], "semantic": semantic[:limit], "fused": fused,
                       "union": union, "visual_budget_control": visual[:2 * limit]},
            "results": union, "selection": "visual_top_k_preserved_plus_semantic_top_k_new_sources",
            "maximum_candidates": 2 * limit, "visual_scan_count": visual_report["scanned_vectors"],
            "query_image_evidence": visual_report["query_evidence"],
            "scores_used_for_ranking": False, "test_samples_consumed": 0,
            "claim": "candidate_similarity_not_failure_cause_or_training_admission", "trainable": False}
