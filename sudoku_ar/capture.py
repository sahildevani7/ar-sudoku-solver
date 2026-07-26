import threading

import cv2


class Camera:
    '''
    cv2.VideoCapture buffers frames, so a processing loop slower than the camera's frame rate
    ends up displaying stale frames even once it speeds back up. This runs the blocking
    cap.read() in a background thread and always hands the main loop the newest frame,
    dropping any it didn't get to.
    '''

    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self._lock:
                self._frame = frame

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        with self._lock:
            frame = self._frame
        return frame is not None, frame

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.cap.release()
