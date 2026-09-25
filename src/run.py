"""CLI entry point: python -m src.run <command> [--frac F]"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("check", "Step 0 data checks"),
                        ("norm", "Step 1 normalization + report")):
        c = sub.add_parser(name, help=help_)
        c.add_argument("--frac", type=float, default=1.0)
    a = p.parse_args()

    from src import evaluate as E
    if a.cmd == "check":
        E.data_check(a.frac)
    elif a.cmd == "norm":
        E.norm_report(a.frac)


if __name__ == "__main__":
    main()
