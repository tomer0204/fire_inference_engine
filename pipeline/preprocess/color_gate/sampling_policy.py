def should_check_color(frame_idx: int, every_n: int) -> bool:
    return frame_idx % every_n == 0

def should_run_yolo(frame_idx: int, suspicious: bool, cooldown_state: dict, cooldown_frames: int) -> bool:
    if cooldown_state["cooldown_left"] > 0:
        cooldown_state["cooldown_left"] -= 1
        return False
    if suspicious:
        cooldown_state["cooldown_left"] = cooldown_frames
        return True
    return False
