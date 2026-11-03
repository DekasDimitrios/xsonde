import os
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xsonde", description="Explainability probes for earth-system models.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="compute every run of a sweep config")
    run.add_argument("config", help="path to the sweep yaml")
    run.add_argument("--force", action="store_true", help="recompute runs already marked done")
    run.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", 0)),
                     help="index of this task, defaults to SLURM_PROCID")
    run.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)),
                     help="number of tasks, defaults to SLURM_NTASKS")
    args = parser.parse_args(argv)

    if args.command == "run":
        from xsonde.runner.run import run_sweep

        run_sweep(args.config, force=args.force, shard=args.shard, nshards=args.nshards)


if __name__ == "__main__":
    main()
