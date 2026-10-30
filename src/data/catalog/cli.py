"""Scene catalog command line; run with the installed DriveRSI environment."""

import argparse
import json
from pathlib import Path

from driversi.data.catalog.observations import file_hash
from driversi.data.catalog.store import Catalog


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("import", "stats", "query", "label", "import-scores", "embed", "neighbors", "plan"))
    parser.add_argument("--database", required=True)
    parser.add_argument("--split", choices=("navtrain", "navtest"), required=True)
    parser.add_argument("--index")
    parser.add_argument("--descriptors")
    parser.add_argument("--purpose", choices=("training", "evaluation"))
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--profile")
    parser.add_argument("--visual-tag", action="append", default=[])
    parser.add_argument("--text")
    parser.add_argument("--config")
    parser.add_argument("--tokens", help="JSON list of tokens from this database only")
    parser.add_argument("--output")
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--score-shard", action="append", default=[])
    parser.add_argument("--score-run")
    parser.add_argument("--score-below", type=float)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--job", choices=("embed", "label"))
    parser.add_argument("--query-token")
    parser.add_argument("--vector-profile")
    parser.add_argument("--shortlist", type=int, default=128)
    parser.add_argument("--include-source", action="store_true")
    args = parser.parse_args()
    catalog = Catalog(args.database, args.split)
    try:
        if args.command == "import":
            from driversi_agents.chainflow.benchmark import read_index

            index = read_index(args.index, args.split)
            desc = json.loads(Path(args.descriptors).read_text())
            catalog.ingest(index["records"], desc.get("records", desc.get("rows")), {
                "index": str(Path(args.index).resolve()), "split": args.split,
                "index_sha256": file_hash(args.index),
                "descriptor_sha256": file_hash(args.descriptors),
                "official_filter_sha256": index["official_filter_sha256"],
                "complete": True,
            })
            result = catalog.stats()
        elif args.command == "query":
            result = catalog.query(args.purpose, args.tag, args.limit, profile=args.profile,
                                   visual_tags=args.visual_tag, text=args.text,
                                   score_run=args.score_run, score_below=args.score_below)
        elif args.command == "import-scores":
            from driversi.rsi.decisions.train_pool import read_scored_rows

            rows, provenance = read_scored_rows(args.score_shard)
            if provenance is None:
                raise ValueError("No identified score shards supplied")
            run = catalog.import_train_scores(rows, provenance)
            result = {"score_run": run, "stats": catalog.stats()}
        elif args.command == "plan":
            from driversi.data.catalog.batches import plan_batches

            result = plan_batches(catalog, args.purpose, args.config, args.output,
                                  args.job, args.batch_size)
        elif args.command == "embed":
            from driversi.data.catalog.features import dump_batch

            result = dump_batch(catalog, json.loads(Path(args.config).read_text()),
                                json.loads(Path(args.tokens).read_text()), args.output,
                                max_scenes=args.limit, max_seconds=args.max_seconds)
        elif args.command == "neighbors":
            from driversi.data.catalog.vectors import VectorStore

            result = VectorStore(catalog).search(
                args.purpose, args.query_token, args.vector_profile, limit=args.limit,
                shortlist=args.shortlist, exclude_source=not args.include_source,
                filters={"tags": args.tag, "profile": args.profile, "visual_tags": args.visual_tag,
                         "text": args.text, "score_run": args.score_run, "score_below": args.score_below},
            )
            if args.output:
                Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "label":
            from driversi.data.catalog.labeling import label_batch

            result = label_batch(catalog, json.loads(Path(args.config).read_text()),
                                 json.loads(Path(args.tokens).read_text()), args.output,
                                 max_scenes=args.limit, max_seconds=args.max_seconds)
        else:
            result = catalog.stats()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        catalog.close()


if __name__ == "__main__":
    main()
