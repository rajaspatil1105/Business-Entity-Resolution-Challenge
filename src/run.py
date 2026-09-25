"""CLI: python -m src.run <check|norm|block|feat|stagea> [--frac F] [--qfrac Q] [--split train|test]"""
import argparse

from src import evaluate as E


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["check", "norm", "block", "feat", "stagea"])
    p.add_argument("--frac", type=float, default=1.0, help="record sample of the split")
    p.add_argument("--qfrac", type=float, default=1.0, help="S1 query sample (full pool searched)")
    p.add_argument("--split", default="train", choices=["train", "test"])
    a = p.parse_args()
    if a.cmd == "check":
        E.data_check(a.frac)
    elif a.cmd == "norm":
        E.norm_report(a.frac)
    elif a.cmd == "block":
        if a.split == "train":
            E.block_report(a.frac, a.qfrac)
        else:
            from src import blocking as B
            print(f"test candidates: {B.block_split('test', a.frac, a.qfrac).height:,}")
    elif a.cmd == "feat":
        if a.split == "train":
            E.feat_report(a.frac, a.qfrac)
        else:
            from src import features as F
            print(f"test feature rows: {F.build('test', a.frac, a.qfrac).height:,}")
    elif a.cmd == "stagea":
        E.stagea_report(a.frac, a.qfrac)


if __name__ == "__main__":
    main()
