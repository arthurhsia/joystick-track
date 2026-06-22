"""
ECoG joystick cursor decoding — main entry point.

Usage:
  python main.py              # run both ridge and hybrid pipelines
  python main.py --ridge      # ridge only
  python main.py --hybrid     # hybrid only
"""

import argparse
from pipeline.ridge_pipeline  import run_ridge
from pipeline.hybrid_pipeline import run_hybrid


def main():
    parser = argparse.ArgumentParser(description='ECoG joystick decoding pipeline')
    parser.add_argument('--ridge',  action='store_true', help='Run ridge pipeline only')
    parser.add_argument('--hybrid', action='store_true', help='Run hybrid pipeline only')
    args = parser.parse_args()
    run_both = not args.ridge and not args.hybrid

    if args.ridge  or run_both: run_ridge()
    if args.hybrid or run_both: run_hybrid()


if __name__ == '__main__':
    main()
