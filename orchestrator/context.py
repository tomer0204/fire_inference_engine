class OrchestratorContext:
    def __init__(self, camera_id, run_id, fps):
        self.camera_id = camera_id
        self.run_id = run_id
        self.fps = fps

        self.frame_index = 0
        self.active_rois = []
        self.temporal_buffer = []
        self.current_state = "COLOR_GATE"
