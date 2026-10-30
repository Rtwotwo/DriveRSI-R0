"""SQLite scene facts and versioned, unverified visual observations."""

import hashlib
import json
import math
import sqlite3
from pathlib import Path


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def identity(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


class Catalog:
    def __init__(self, path, split, *, readonly=False):
        if split not in ("navtrain", "navtest"):
            raise ValueError("Catalog split must be navtrain or navtest")
        self.path, self.split = Path(path), split
        if readonly:
            self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
            self.db.row_factory = sqlite3.Row
            row = self.db.execute("SELECT value FROM metadata WHERE key='split'").fetchone()
            if row is None or row[0] != split:
                self.db.close()
                raise ValueError("Database split differs; cross-split catalog use is forbidden")
            from .retrieval_collection import activate_registered_collection
            try:
                activate_registered_collection(self)
            except BaseException:
                self.db.close()
                raise
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS scenes (
                token TEXT PRIMARY KEY, source_log TEXT NOT NULL, source_group TEXT NOT NULL,
                record TEXT NOT NULL, descriptor TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS scene_source ON scenes(source_group);
            CREATE TABLE IF NOT EXISTS tags (
                token TEXT REFERENCES scenes(token), tag TEXT, PRIMARY KEY(token,tag));
            CREATE INDEX IF NOT EXISTS tag_lookup ON tags(tag);
            CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, config TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, token TEXT REFERENCES scenes(token),
                profile TEXT REFERENCES profiles(id), input_id TEXT NOT NULL,
                status TEXT NOT NULL, observation TEXT, evidence TEXT NOT NULL,
                raw TEXT, error TEXT, cost TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS label_lookup ON attempts(token,profile,input_id,status);
            CREATE INDEX IF NOT EXISTS profile_lookup ON attempts(profile,status,token);
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY, attempt INTEGER REFERENCES attempts(id),
                reviewer TEXT NOT NULL, verdict TEXT NOT NULL, note TEXT NOT NULL,
                created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS review_lookup ON reviews(attempt,id);
            CREATE TABLE IF NOT EXISTS visual_tags (
                attempt INTEGER REFERENCES attempts(id), tag TEXT,
                PRIMARY KEY(attempt,tag));
            CREATE VIRTUAL TABLE IF NOT EXISTS captions USING fts5(caption);
            CREATE TABLE IF NOT EXISTS score_runs (id TEXT PRIMARY KEY, provenance TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS scores (
                run TEXT REFERENCES score_runs(id), token TEXT REFERENCES scenes(token),
                valid INTEGER NOT NULL, score REAL, metrics TEXT NOT NULL,
                PRIMARY KEY(run,token));
            CREATE INDEX IF NOT EXISTS score_lookup ON scores(run,valid,score,token);
        """)
        old = self.db.execute("SELECT value FROM metadata WHERE key='split'").fetchone()
        if old and old[0] != split:
            self.db.close()
            raise ValueError("Database split differs; cross-split catalog use is forbidden")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('split',?)", (split,))
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('schema_version','1')")

    def close(self):
        self.db.close()

    def ingest(self, records, descriptors, provenance):
        by_token = {r["token"]: r for r in descriptors}
        if len(by_token) != len(descriptors):
            raise ValueError("Duplicate descriptor token")
        descriptors = by_token
        if len({r["token"] for r in records}) != len(records):
            raise ValueError("Duplicate scene token")
        if set(descriptors) != {r["token"] for r in records}:
            raise ValueError("Descriptor population differs from scene population")
        with self.db:
            for row in records:
                token = row["token"]
                desc = descriptors[token]
                if row["kind"] != "original_" + self.split:
                    raise ValueError("Source record split mismatch")
                if desc["source_log"] != row["source_log"]:
                    raise ValueError("Descriptor source mismatch")
                values = (token, row["source_log"], desc["source_group"], encode(row), encode(desc))
                old = self.db.execute("SELECT * FROM scenes WHERE token=?", (token,)).fetchone()
                if old and tuple(old) != values:
                    raise ValueError("Existing scene changed: use a new catalog version")
                self.db.execute("INSERT OR IGNORE INTO scenes VALUES (?,?,?,?,?)", values)
                self.db.executemany(
                    "INSERT OR IGNORE INTO tags VALUES (?,?)",
                    ((token, tag) for tag in desc.get("conditions", [])),
                )
            value = encode(provenance)
            old = self.db.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
            if old and old[0] != value:
                raise ValueError("Catalog population provenance changed")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('source',?)", (value,))

    def scene(self, token):
        row = self.db.execute("SELECT * FROM scenes WHERE token=?", (token,)).fetchone()
        if not row:
            raise ValueError("Unknown scene token")
        return {"record": json.loads(row["record"]), "descriptor": json.loads(row["descriptor"])}

    def query(self, purpose, tags=(), limit=20, *, profile=None, visual_tags=(), text=None,
              score_run=None, score_below=None):
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("Query limit must be between 1 and 10000")
        sql, params = self.token_query(purpose, tags, limit, profile=profile,
                                      visual_tags=visual_tags, text=text,
                                      score_run=score_run, score_below=score_below)
        return [self.scene(r[0]) for r in self.db.execute(sql, params).fetchall()]

    def token_query(self, purpose, tags=(), limit=None, *, profile=None, visual_tags=(), text=None,
                    score_run=None, score_below=None):
        if purpose not in ("training", "evaluation"):
            raise ValueError("Explicit query purpose required")
        if (purpose == "training") != (self.split == "navtrain"):
            raise ValueError("Query purpose and database split disagree")
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 10000):
            raise ValueError("Query limit must be between 1 and 10000")
        clauses = ["s.token IN (SELECT token FROM tags WHERE tag=?)"] * len(tags)
        params = list(tags)
        if score_below is not None:
            if purpose != "training" or not score_run or not 0 <= score_below <= 1:
                raise ValueError("Training score query requires an explicit run and normalized threshold")
            if not self.db.execute("SELECT 1 FROM score_runs WHERE id=?", (score_run,)).fetchone():
                raise ValueError("Unknown score run")
            clauses.append("s.token IN (SELECT token FROM scores WHERE run=? AND valid=1 AND score<?)")
            params.extend((score_run, score_below))
        if visual_tags or text:
            if not profile or not self.db.execute("SELECT 1 FROM profiles WHERE id=?", (profile,)).fetchone():
                raise ValueError("Visual retrieval requires an explicit existing annotation profile")
            label_clauses = [
                "a.profile=?", "a.status='validated'",
                "a.id=(SELECT max(z.id) FROM attempts z WHERE z.token=a.token AND z.profile=a.profile)",
                "coalesce((SELECT verdict FROM reviews q WHERE q.attempt=a.id ORDER BY id DESC LIMIT 1),'unreviewed')!='rejected'",
            ]
            params.append(profile)
            profile_data=json.loads(self.db.execute("SELECT config FROM profiles WHERE id=?",(profile,)).fetchone()[0])
            if profile_data.get("config",{}).get("prompt_version")=="history_state_v8":
                label_clauses.append("(SELECT verdict FROM reviews q WHERE q.attempt=a.id ORDER BY id DESC LIMIT 1)='spot_check_pass'")
            for tag in visual_tags:
                label_clauses.append("EXISTS (SELECT 1 FROM visual_tags v WHERE v.attempt=a.id AND v.tag=?)")
                params.append(tag)
            if text:
                # Literal phrase search: captions never become SQL or FTS operators.
                label_clauses.append("a.id IN (SELECT rowid FROM captions WHERE captions MATCH ?)")
                params.append('"' + text.replace('"', '""') + '"')
            clauses.append("s.token IN (SELECT a.token FROM attempts a WHERE " + " AND ".join(label_clauses) + ")")
        sql = "SELECT s.token FROM scenes s"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY s.token"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return sql, params

    def import_train_scores(self, rows, provenance):
        if self.split != "navtrain" or provenance.get("split") != "navtrain" or (
            provenance.get("purpose") != "training_pool_mining"
        ):
            raise ValueError("Only training-owned measurements can enter the training score table")
        source = self.db.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
        if source is None or json.loads(source[0]).get("index_sha256") != provenance.get("index_sha256"):
            raise ValueError("Scores and catalog use different population indexes")
        run = identity(provenance)
        fields = ("score", "no_at_fault_collisions", "drivable_area_compliance",
                  "driving_direction_compliance", "ego_progress", "time_to_collision_within_bound", "comfort")
        # One sequential scan avoids repeated random reads of large scene records.
        source_logs = dict(self.db.execute("SELECT token,source_log FROM scenes"))
        pending = []
        for row in rows:
            source = source_logs.get(row["token"])
            if source is None or row["log_name"] != source or type(row["valid"]) is not bool:
                raise ValueError("Score source/valid flag mismatch")
            metrics = {k: row[k] for k in fields} if row["valid"] else {}
            if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
                   for v in metrics.values()):
                raise ValueError("Normalized score metrics required")
            pending.append((run, row["token"], int(row["valid"]), metrics.get("score"), encode(metrics)))
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO score_runs VALUES (?,?)", (run, encode(provenance)))
        # Append-only batches avoid holding the single SQLite writer while the
        # whole population is decoded. Interrupted imports can be repeated.
        for start in range(0, len(pending), 256):
            with self.db:
                for values in pending[start:start + 256]:
                    old = self.db.execute("SELECT * FROM scores WHERE run=? AND token=?", values[:2]).fetchone()
                    if old and tuple(old) != values:
                        raise ValueError("A measured score changed within the same run")
                    self.db.execute("INSERT OR IGNORE INTO scores VALUES (?,?,?,?,?)", values)
        return run

    def profile(self, config):
        key = identity(config)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO profiles VALUES (?,?)", (key, encode(config)))
        return key

    def completed(self, token, profile, input_id):
        row = self.db.execute(
            "SELECT input_id,status FROM attempts WHERE token=? AND profile=? ORDER BY id DESC LIMIT 1",
            (token, profile),
        ).fetchone()
        return row is not None and tuple(row) == (input_id, "validated")

    def review(self, attempt, reviewer, verdict, note):
        if verdict not in ("rejected", "spot_check_pass") or not reviewer or not note:
            raise ValueError("Explicit reviewer, verdict and rationale required")
        with self.db:
            self.db.execute("INSERT INTO reviews(attempt,reviewer,verdict,note) VALUES (?,?,?,?)",
                            (attempt, reviewer, verdict, note))

    def annotation(self, token, profile):
        row = self.db.execute(
            "SELECT id,input_id,status,observation,evidence FROM attempts "
            "WHERE token=? AND profile=? ORDER BY id DESC LIMIT 1", (token, profile),
        ).fetchone()
        if row is None or row["status"] != "validated":
            return None
        review = self.db.execute("SELECT reviewer,verdict,note FROM reviews WHERE attempt=? ORDER BY id DESC LIMIT 1", (row["id"],)).fetchone()
        if review and review["verdict"] == "rejected":
            return None
        return {**dict(row), "observation": json.loads(row["observation"]),
                "evidence": json.loads(row["evidence"]), "review": dict(review) if review else None}

    def record_attempt(self, token, profile, evidence, raw, cost, *, observation=None, error=None):
        from .observations import validate_observation

        if observation is not None:
            observation = validate_observation(observation, evidence)
        status = "validated" if observation is not None else "failed"
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO attempts(token,profile,input_id,status,observation,evidence,raw,error,cost) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (token, profile, identity(evidence), status, encode(observation), encode(evidence),
                 raw, error, encode(cost)),
            )
            if observation is not None:
                self.db.execute("INSERT INTO captions(rowid,caption) VALUES (?,?)",
                                (cursor.lastrowid, observation["caption"]))
                self.db.executemany("INSERT INTO visual_tags VALUES (?,?)",
                                    ((cursor.lastrowid, row["tag"]) for row in observation["observations"]))
        return cursor.lastrowid

    def stats(self):
        return {
            "split": self.split,
            "scenes": self.db.execute("SELECT count(*) FROM scenes").fetchone()[0],
            "source_groups": self.db.execute("SELECT count(DISTINCT source_group) FROM scenes").fetchone()[0],
            "attempts": dict(self.db.execute("SELECT status,count(*) FROM attempts GROUP BY status")),
            "labeled_scenes": self.db.execute("SELECT count(DISTINCT token) FROM attempts WHERE status='validated'").fetchone()[0],
            "score_rows": self.db.execute("SELECT count(*) FROM scores").fetchone()[0],
            "visual_reviews": dict(self.db.execute("SELECT verdict,count(*) FROM reviews GROUP BY verdict")),
            "annotation_scope": "unverified_history_image_observation",
            "trainable": False,
        }
