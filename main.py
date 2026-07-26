import argparse

from sudoku_ar.pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(description="AR Sudoku Solver")
    parser.add_argument("--source", default=None,
                         help="Path to a video/image file to use instead of the live webcam")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(source=args.source)
