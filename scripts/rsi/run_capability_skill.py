"""Execute one capability-selected NAVSIM teacher skill; never score navtrain students."""
import argparse
import json

from driversi.rsi.experiments.skill_execution import run_assignment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("plan","assignment","database","output","memory","maps-root","python","checkpoint"):
        parser.add_argument("--"+key,required=True)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    report=run_assignment(args.plan,args.assignment,args.database,args.output,args.memory,
        maps_root=args.maps_root,python=args.python,checkpoint=args.checkpoint,device=args.device)
    print(json.dumps({k:report.get(k) for k in ("status","assignment_id","wall_seconds","training_executed")}))


if __name__ == "__main__":main()
