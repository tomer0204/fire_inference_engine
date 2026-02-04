from ingest.stream_reader import StreamReader
from ingest.clock import Clock
from ingest.s3_minio_reader import S3MinioWriter
from pipeline.preprocess.color_gate import color_gate

def run_live(camera_id, run_id, video_path, fps):
    reader = StreamReader(video_path)
    clock = Clock(fps)

    writer = S3MinioWriter(
        bucket="fire-frames",
        endpoint="http://localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin"
    )

    frame_idx = 0

    while True:
        frame = reader.read_frame()
        if frame is None:
            break

        rois = color_gate(frame)

        key = f"{camera_id}/{run_id}/frame_{frame_idx:06d}.jpg"
        writer.write_frame(frame, key)

        frame_idx += 1
        clock.tick()

    reader.close()
