use crate::io::{Detection, Detections, OutputMeta, PostprocessConfig, RawOutput};
use anyhow::Result;

pub fn run(raw: &RawOutput, cfg: &PostprocessConfig) -> Result<Detections> {
    let mut dets = decode_nx6(raw, cfg.conf_threshold)?;
    dets = nms(dets, cfg.nms_iou_threshold, cfg.max_detections);
    Ok(Detections {
        detections: dets,
        meta: OutputMeta {
            input_width: 0,
            input_height: 0,
            model_width: 0,
            model_height: 0,
        },
    })
}

fn decode_nx6(raw: &RawOutput, conf: f32) -> Result<Vec<Detection>> {
    if raw.shape.len() != 2 || raw.shape[1] != 6 {
        anyhow::bail!("expected Nx6 output, got shape {:?}", raw.shape);
    }
    let n = raw.shape[0];
    if raw.data.len() != n * 6 {
        anyhow::bail!("output data length mismatch");
    }

    let mut out = Vec::new();
    for i in 0..n {
        let base = i * 6;
        let x1 = raw.data[base];
        let y1 = raw.data[base + 1];
        let x2 = raw.data[base + 2];
        let y2 = raw.data[base + 3];
        let score = raw.data[base + 4];
        let class_id = raw.data[base + 5] as i32;
        if score >= conf {
            out.push(Detection { x1, y1, x2, y2, score, class_id });
        }
    }
    Ok(out)
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let x_left = a.x1.max(b.x1);
    let y_top = a.y1.max(b.y1);
    let x_right = a.x2.min(b.x2);
    let y_bottom = a.y2.min(b.y2);

    let inter_w = (x_right - x_left).max(0.0);
    let inter_h = (y_bottom - y_top).max(0.0);
    let inter = inter_w * inter_h;

    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);

    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn nms(mut dets: Vec<Detection>, iou_th: f32, max_det: usize) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep: Vec<Detection> = Vec::new();

    for d in dets.into_iter() {
        let mut suppressed = false;
        for k in keep.iter() {
            if iou(&d, k) >= iou_th {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            keep.push(d);
            if keep.len() >= max_det {
                break;
            }
        }
    }

    keep
}
