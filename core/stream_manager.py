import threading
import time
import uuid
import cv2
from ingest.stream_reader import StreamReader


class StreamManager:
    def __init__(self, cameras: dict, socketio, archive_writer=None, enable_archive=False):
        self.cameras = cameras
        self.socketio = socketio
        self.archive_writer = archive_writer
        self.enable_archive = enable_archive

        self._lock = threading.Lock()
        self._thread = None
        self._stop = False

        self._camera_id = None
        self._run_id = None
        self._prefix = None
        self._fps = 10


    def get_state(self):
        with self._lock:
            active = self._thread is not None and self._thread.is_alive()
            return {
                "active": active,
                "camera_id": self._camera_id,
                "run_id": self._run_id,
                "s3_prefix": self._prefix,
                "fps": self._fps,
            }


    def start(self, camera_id: int, fps: int = 10):
        camera_id = int(camera_id)

        if camera_id not in self.cameras:
            raise ValueError("Camera not found")

        self.stop()

        run_id = uuid.uuid4().hex
        prefix = f"{camera_id}/{run_id}/"
        fps = int(fps)

        if self.enable_archive and self.archive_writer:
            self.archive_writer.ensure_bucket()

        with self._lock:
            self._stop = False
            self._camera_id = camera_id
            self._run_id = run_id
            self._prefix = prefix
            self._fps = fps

        t = threading.Thread(target=self._loop, daemon=True)
        self._thread = t
        t.start()

        return {"run_id": run_id, "s3_prefix": prefix, "fps": fps}

    def stop(self, run_id: str | None = None):
        with self._lock:
            if self._thread is None:
                return
            if run_id and self._run_id != run_id:
                return
            self._stop = True

        self._thread.join(timeout=5)

        with self._lock:
            self._thread = None
            self._stop = False
            self._camera_id = None
            self._run_id = None
            self._prefix = None


    def _loop(self):
        st = self.get_state()
        camera_id = st["camera_id"]
        run_id = st["run_id"]
        prefix = st["s3_prefix"]
        fps = st["fps"]

        cfg = self.cameras[int(camera_id)]
        reader = StreamReader(source_cfg=cfg["source"])

        delay = 1.0 / max(1, fps)
        idx = 0

        try:
            print(f"[STREAM] Started camera {camera_id} run={run_id} fps={fps}")

            while True:
                with self._lock:
                    if self._stop:
                        break

                frame = reader.read()
                if frame is None:
                    break

                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    self.socketio.sleep(delay)
                    continue

                jpeg = buf.tobytes()

               
                self.socketio.emit("infer_frame", jpeg, room=f"run:{run_id}")

                if self.enable_archive and self.archive_writer:
                    try:
                        key = f"{prefix}{idx:06d}.jpg"
                        self.archive_writer.put_jpeg(key, jpeg)
                    except Exception:
                        pass

                idx += 1
                self.socketio.sleep(delay)

        finally:
            reader.close()
            print(f"[STREAM] Stopped camera {camera_id}")
