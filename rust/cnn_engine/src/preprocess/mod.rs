use crate::io::{Frame, PreprocessConfig, Tensor};
use anyhow::Result;
use image::{imageops::FilterType, DynamicImage, GenericImageView};

pub fn run(frame: &Frame, cfg: &PreprocessConfig) -> Result<Tensor> {
    let (mw, mh) = (cfg_resize_target_width(frame, cfg)?, cfg_resize_target_height(frame, cfg)?);
    let resized = frame.image.resize_exact(mw, mh, FilterType::Triangle);

    let (w, h) = resized.dimensions();
    let data = image_to_chw_f32(&resized, cfg)?;

    Ok(Tensor {
        data,
        shape: [1, 3, h as usize, w as usize],
    })
}

fn cfg_resize_target_width(frame: &Frame, cfg: &PreprocessConfig) -> Result<u32> {
    Ok(super::process::model_target::target_width(frame, cfg.resize.keep_aspect))
}

fn cfg_resize_target_height(frame: &Frame, cfg: &PreprocessConfig) -> Result<u32> {
    Ok(super::process::model_target::target_height(frame, cfg.resize.keep_aspect))
}

fn image_to_chw_f32(img: &DynamicImage, cfg: &PreprocessConfig) -> Result<Vec<f32>> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut out = vec![0.0f32; (3 * w * h) as usize];

    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y).0;
            let mut r = p[0] as f32;
            let mut g = p[1] as f32;
            let mut b = p[2] as f32;

            if cfg.normalize.scale_255 {
                r /= 255.0;
                g /= 255.0;
                b /= 255.0;
            }

            r = (r - cfg.normalize.mean[0]) / cfg.normalize.std[0];
            g = (g - cfg.normalize.mean[1]) / cfg.normalize.std[1];
            b = (b - cfg.normalize.mean[2]) / cfg.normalize.std[2];

            let idx = (y * w + x) as usize;
            out[idx] = r;
            out[(w * h) as usize + idx] = g;
            out[(2 * w * h) as usize + idx] = b;
        }
    }

    Ok(out)
}
