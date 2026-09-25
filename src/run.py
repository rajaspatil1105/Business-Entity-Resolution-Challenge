"""CLI entry point: python -m src.run <command>"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="Step 0 data checks")
    c.add_argument("--frac", type=float, default=1.0)
    a = p.parse_args()

    if a.cmd == "check":
        from src import evaluate as E
        E.data_check(a.frac)


if __name__ == "__main__":
    main()
