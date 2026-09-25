"""CLI entry point: python -m src.run <command> [options]"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("check", "Step 0 data checks"),
                        ("norm", "Step 1 normalization + report"),
                        ("block", "Step 2 blocking (+ report on train)")):
        c = sub.add_parser(name, help=help_)
        c.add_argument("--frac", type=float, default=1.0, help="data sample share")
        if name == "block":
            c.add_argument("--qfrac", type=float, default=1.0, help="share of S1 used as queries")
            c.add_argument("--split", default="train", choices=["train", "test"])
    a = p.parse_args()

    from src import evaluate as E
    if a.cmd == "check":
        E.data_check(a.frac)
    elif a.cmd == "norm":
        E.norm_report(a.frac)
    elif a.cmd == "block":
        if a.split == "train":
            E.block_report(a.frac, a.qfrac)
        else:
            from src import blocking as B
            B.block_split("test", a.frac, a.qfrac)


if __name__ == "__main__":
    main()
