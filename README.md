# AR Sudoku Solver

An Augmented Reality Sudoku solver that uses computer vision and machine learning to solve Sudoku puzzles in real-time. The application captures a Sudoku puzzle using a webcam, processes it using OpenCV, and overlays the solution onto the original image.

https://github.com/user-attachments/assets/340471a7-90d5-4bc5-a00f-66f15f9fc6b2

## How It Works

1. **Grid Detection**
   - The application uses contour detection to identify the Sudoku grid
   - Perspective transformation is applied to get a top-down view
   - Grid validation ensures accurate detection

2. **Digit Recognition**
   - The grid is split into 81 individual cells
   - Each cell is processed and fed into the CNN model
   - The model predicts the digit in each cell

3. **Sudoku Solving**
   - The recognized digits form the initial puzzle
   - Exact cover algorithm solves the puzzle
   - Solution validation ensures correctness

4. **AR Overlay**
   - The solution is overlaid onto the original image
   - Perspective transformation ensures accurate placement
   - Visual feedback shows the solving process

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

Point a webcam at a Sudoku puzzle and hold it steady. Once a read is confirmed across a few
consecutive frames, the solution locks onto the grid and tracks it until the puzzle is taken out
of view.

**Controls:**
- `q` - quit
- `r` - reset (release the current lock and re-scan)

To run against a video or image file instead of a live webcam (useful for testing without a
camera):

```bash
python main.py --source path/to/video.mp4
```

## Testing

```bash
pip install -r requirements.txt  # includes pytest
pytest tests/
```

Covers the solver, validator, grid detection, digit recognition, and the acquisition state
machine - all runnable without a camera.

## Architecture

```
sudoku_ar/
  config.py       # tunables (warp size, thresholds, state-machine timing)
  capture.py      # threaded camera reader - always serves the newest frame
  detector.py     # grid contour/corner detection, perspective warp
  recognizer.py   # DigitRecognizer - batched CNN inference over all 81 cells
  solver.py       # exact-cover sudoku solver
  validator.py    # sudoku rule validation
  overlay.py      # renders the solution and composites it onto the live frame
  hud.py          # on-screen instructions/status/FPS text
  pipeline.py     # SEARCHING -> CONFIRMING -> SOLVED acquisition state machine
main.py           # entrypoint (CLI args, calls sudoku_ar.pipeline.run)
tests/
```

A puzzle is only accepted once the same digit read repeats across several consecutive frames,
and only locked in once the solver actually finds a solution - a single noisy frame can't lock in
a wrong answer. Once locked, digit recognition stops entirely; only the grid's corners are
tracked each frame to keep the overlay in place, until the puzzle is removed or `r` is pressed.
