use anyhow::Result;
use image::{DynamicImage, GenericImageView};

#[derive(Clone)]
pub struct Frame {
    pub image: DynamicImage,
    pub width: u32,
    pub height: u32,
}

impl Frame {
    pub fn from_path(path: &str) -> Result<Self> {
        let image = image::open(path)?;
        let (width, height) = image.dimensions();
        Ok(Self { image, width, height })
    }
}

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: [usize; 4],
}

#[derive(Clone)]
pub struct RawOutput {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Detections {
    pub detections: Vec<Detection>,
    pub meta: OutputMeta,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutputMeta {
    pub input_width: u32,
    pub input_height: u32,
    pub model_width: u32,
    pub model_height: u32,
}
