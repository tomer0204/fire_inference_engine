import cv2
import json
import yaml
import requests
from minio import Minio
import io

from ingest.stream_reader import StreamReader
from pipeline.preprocess.color_gate.ycrcb_gate import ycrcb_fire_gate
from pipeline.preprocess.color_gate.sampling_policy import should_check_color, should_run_yolo
from queue.frame_queue import FrameQueue
from queue.detection_queue import DetectionQueue
from queue.temporal_policy import temporal_decision

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def encode_jpg(frame_bgr, quality=90):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("failed to encode jpg")
    return buf.tobytes()

def ensure_bucket(mc, bucket):
    if not mc.bucket_exists(bucket):
        mc.make_bucket(bucket)

def put_bytes(mc, bucket, key, data, content_type):
    mc.put_object(bucket, key, io.BytesIO(data), length=len(data), content_type=content_type)

def main():
    cfg = load_cfg("configs/live.yaml")

    video_path = cfg["video"]["path"]
    rust_url = cfg["rust"]["infer_url"]
    rust_timeout = cfg["rust"]["timeout_sec"]

    mc = Minio(
        cfg["minio"]["endpoint"],
        access_key=cfg["minio"]["access_key"],
        secret_key=cfg["minio"]["secret_key"],
        secure=cfg["minio"]["secure"],
    )
    frames_bucket = cfg["minio"]["frames_bucket"]
    dets_bucket = cfg["minio"]["detections_bucket"]
    ensure_bucket(mc, frames_bucket)
    ensure_bucket(mc, dets_bucket)

    frame_q = FrameQueue(maxlen=int(cfg["queues"]["frame_queue_maxlen"]))
    det_q = DetectionQueue(maxlen=int(cfg["queues"]["det_queue_maxlen"]))

    reader = StreamReader(video_path)
    cooldown = {"cooldown_left": 0}

    every_n = int(cfg["sampling"]["check_color_every_n"])
    cooldown_frames = int(cfg["sampling"]["yolo_cooldown_frames"])

    t_win = int(cfg["queues"]["temporal_window_t"])
    m_req = int(cfg["queues"]["temporal_require_m"])
    min_score = float(cfg["temporal"]["min_score"])

    print("Starting live MVP pipeline...")
    try:
        while True:
            frame = reader.read()
            if frame is None:
                print("End of video.")
                break

            frame_idx = reader.frame_idx
            frame_meta = {"frame_idx": frame_idx}
            frame_q.push(frame_idx, frame, frame_meta)

            suspicious = False
            ratio = 0.0
            rois = []

            if should_check_color(frame_idx, every_n=every_n):
                suspicious, ratio, rois = ycrcb_fire_gate(frame, cfg["color_gate"])

            run_yolo = should_run_yolo(
                frame_idx=frame_idx,
                suspicious=suspicious,
                cooldown_state=cooldown,
                cooldown_frames=cooldown_frames
            )

            if not run_yolo:
                if frame_idx % 30 == 0:
                    print(f"[{frame_idx}] skip yolo suspicious={suspicious} ratio={ratio:.4f} rois={len(rois)}")
                continue

            jpg = encode_jpg(frame, quality=cfg["video"]["jpg_quality"])
            frame_key = f"frames/frame_{frame_idx:06d}.jpg"
            put_bytes(mc, frames_bucket, frame_key, jpg, "image/jpeg")

            r = requests.post(
                rust_url,
                data=jpg,
                headers={"Content-Type": "image/jpeg"},
                timeout=rust_timeout
            )
            if r.status_code != 200:
                print("Rust infer failed:", r.status_code, r.text)
                continue

            dets = r.json()
            dets["frame_id"] = frame_idx
            dets["frame_key"] = frame_key
            dets["color_gate_ratio"] = ratio
            dets["color_gate_rois"] = rois

            det_q.push(frame_idx, dets)

            win = det_q.window(t_win)
            ok, ok_ratio = temporal_decision(win, min_score=min_score, require_m_of_t=(m_req, t_win))
            dets["temporal_ok"] = ok
            dets["temporal_ratio"] = ok_ratio
            dets["temporal_rule"] = {"m": m_req, "t": t_win, "min_score": min_score}

            det_key = f"detections/frame_{frame_idx:06d}.json"
            put_bytes(mc, dets_bucket, det_key, json.dumps(dets).encode("utf-8"), "application/json")

            print(f"[{frame_idx}] color_ratio={ratio:.4f} yolo_dets={len(dets.get('detections', []))} temporal_ok={ok} saved={det_key}")

    finally:
        reader.close()

if __name__ == "__main__":
    main()
