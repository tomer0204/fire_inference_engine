import os
import cv2

class StreamReader:
    def __init__(self, source_cfg: dict):
        if source_cfg.get("type") != "video":
            raise NotImplementedError("Only video source supported in MVP")

        path = source_cfg["path"]

        if not os.path.exists(path):
            raise RuntimeError(f"Video path does not exist: {path}")

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self):
        if self.cap:
            self.cap.release()
