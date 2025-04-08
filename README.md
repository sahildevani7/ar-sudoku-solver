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
