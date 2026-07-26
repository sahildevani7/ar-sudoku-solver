import cv2

INSTRUCTIONS = [
    "Instructions:",
    "1. Place sudoku grid in camera view",
    "2. Ensure grid is well-lit and clear",
    "3. Hold camera steady for processing",
    "Press 'q' to quit",
]


def draw_instructions(frame):
    y_pos = 30
    for line in INSTRUCTIONS:
        cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += 30


def draw_status(frame, text, color):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
    xloc = frame_width // 2 - text_width // 2
    cv2.putText(frame, text, (xloc, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_fps(frame, fps):
    frame_width = frame.shape[1]
    cv2.putText(frame, f"FPS: {fps}", (frame_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
