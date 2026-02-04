use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub input: InputConfig,
    pub model: ModelConfig,
    pub backend: BackendConfig,
    pub preprocess: PreprocessConfig,
    pub postprocess: PostprocessConfig,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct InputConfig {
    pub source: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub input_width: u32,
    pub input_height: u32,
    pub output_format: OutputFormat,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Nx6,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    pub kind: crate::process::backend::BackendKind,
    pub triton: Option<TritonConfig>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TritonConfig {
    pub url: String,
    pub model_name: String,
    pub model_version: Option<String>,
    pub input_name: String,
    pub output_name: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    pub normalize: NormalizeConfig,
    pub resize: ResizeConfig,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ResizeConfig {
    pub keep_aspect: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NormalizeConfig {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub scale_255: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PostprocessConfig {
    pub conf_threshold: f32,
    pub nms_iou_threshold: f32,
    pub max_detections: usize,
}

impl EngineConfig {
    pub fn from_yaml(path: &str) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Ok(serde_yaml::from_str(&s)?)
    }
}
