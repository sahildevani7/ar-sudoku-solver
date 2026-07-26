import argparse

from sudoku_ar.pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(description="AR Sudoku Solver")
    parser.add_argument("--source", default=None,
                         help="Path to a video/image file to use instead of the live webcam")
    parser.add_argument("--debug", action="store_true",
                         help="Print per-frame acquisition diagnostics (confidence, rejected reads, mismatches) to stderr")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(source=args.source, debug=args.debug)
