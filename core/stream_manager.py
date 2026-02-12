import time
import uuid
import threading
import requests
import cv2

from ingest.stream_reader import StreamReader

class StreamManager:
    def __init__(self, cameras: dict, socketio, archive_writer=None, enable_archive=True):
        self.cameras = cameras
        self.socketio = socketio
        self.archive_writer = archive_writer
        self.enable_archive = enable_archive

        self._lock = threading.Lock()
        self._threads = {}
        self._stop_flags = {}
        self._active = {}

        self.sample_every_n = int(self._env("COLOR_SAMPLE_EVERY_N", "10"))
        self.jpeg_quality = int(self._env("JPEG_QUALITY", "85"))
        self.rust_url = self._env("RUST_INFER_URL", "").strip()
        self.rust_timeout = float(self._env("RUST_TIMEOUT_SEC", "2.0"))

    def _env(self, k, default):
        import os
        return os.getenv(k, default)

    def get_state(self):
        with self._lock:
            return {"active_runs": list(self._active.keys())}

    def start(self, camera_id: int, fps: int = 10):
        camera_id = int(camera_id)
        if camera_id not in self.cameras:
            raise ValueError("Camera not found")

        run_id = uuid.uuid4().hex
        stop_flag = threading.Event()
        s3_prefix = f"cameras/{camera_id}/runs/{run_id}"

        with self._lock:
            self._stop_flags[run_id] = stop_flag
            self._active[run_id] = {
                "camera_id": camera_id,
                "fps": int(fps),
                "started_at": time.time(),
                "frame_index": 0,
                "s3_prefix": s3_prefix
            }

        t = threading.Thread(target=self._run_loop, args=(run_id,), daemon=True)
        with self._lock:
            self._threads[run_id] = t
        t.start()

        return {"run_id": run_id, "camera_id": camera_id, "fps": int(fps), "s3_prefix": s3_prefix}

    def stop(self, run_id: str | None):
        if not run_id:
            return
        run_id = str(run_id)

        with self._lock:
            flag = self._stop_flags.get(run_id)
            t = self._threads.get(run_id)

        if flag:
            flag.set()

        if t and t is not threading.current_thread():
            t.join(timeout=5)

        with self._lock:
            self._threads.pop(run_id, None)
            self._stop_flags.pop(run_id, None)
            self._active.pop(run_id, None)

    def _encode_jpeg(self, frame_bgr):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        if not ok:
            return None
        return buf.tobytes()

    def _maybe_rust_infer(self, jpeg_bytes: bytes):
        if not self.rust_url:
            return None
        try:
            r = requests.post(
                self.rust_url,
                data=jpeg_bytes,
                headers={"Content-Type": "image/jpeg"},
                timeout=self.rust_timeout
            )
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def _run_loop(self, run_id: str):
        with self._lock:
            st = self._active.get(run_id)
            flag = self._stop_flags.get(run_id)
        if not st or not flag:
            return

        camera_id = int(st["camera_id"])
        fps = int(st["fps"])
        delay = 1.0 / max(1, fps)

        cfg = self.cameras[camera_id]
        reader = StreamReader(cfg["source"])

        if self.enable_archive and self.archive_writer:
            self.archive_writer.ensure_bucket()

        last_status_emit = 0.0

        try:
            while not flag.is_set():
                item = reader.read()
                if item is None:
                    break

                frame_idx, frame = item
                do_sample = (frame_idx % max(1, self.sample_every_n) == 0)

                if do_sample:
                    jpeg = self._encode_jpeg(frame)
                    if jpeg:
                        self.socketio.emit(
                            "infer_frame",
                            {"run_id": run_id, "frame_index": frame_idx, "frame": jpeg},
                            room=f"run:{run_id}"
                        )

                        dets = self._maybe_rust_infer(jpeg)
                        if dets is not None:
                            payload = {
                                "run_id": run_id,
                                "frame_index": frame_idx,
                                "detections": dets.get("detections", dets)
                            }
                            self.socketio.emit("infer_result", payload, room=f"run:{run_id}")

                        if self.enable_archive and self.archive_writer:
                            with self._lock:
                                s3_prefix = self._active.get(run_id, {}).get("s3_prefix")
                            if s3_prefix:
                                key = f"{s3_prefix}/frames/{frame_idx:06d}.jpg"
                                self.archive_writer.put_jpeg(key, jpeg)

                now = time.time()
                if now - last_status_emit > 2.0:
                    self.socketio.emit(
                        "infer_status",
                        {"run_id": run_id, "camera_id": camera_id, "frame_index": frame_idx},
                        room=f"run:{run_id}"
                    )
                    last_status_emit = now

                time.sleep(delay)

        finally:
            reader.close()
            self.socketio.emit("infer_ended", {"run_id": run_id}, room=f"run:{run_id}")
            self.stop(run_id)
