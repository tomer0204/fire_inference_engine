use crate::io::{BackendConfig, ModelConfig, RawOutput, Tensor};
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub enum BackendKind {
    Dummy,
    Triton,
}

#[async_trait::async_trait]
pub trait YoloBackend: Send + Sync {
    async fn infer(&self, input: &Tensor) -> Result<RawOutput>;
}

pub struct DummyBackend {
    model: ModelConfig,
}

impl DummyBackend {
    pub fn new(model: ModelConfig) -> Self {
        Self { model }
    }
}

#[async_trait::async_trait]
impl YoloBackend for DummyBackend {
    async fn infer(&self, _input: &Tensor) -> Result<RawOutput> {
        let n = 10usize;
        let mut data = Vec::with_capacity(n * 6);
        for i in 0..n {
            let x1 = 0.1 + i as f32 * 0.01;
            let y1 = 0.1 + i as f32 * 0.01;
            let x2 = 0.4 + i as f32 * 0.01;
            let y2 = 0.4 + i as f32 * 0.01;
            let score = 0.9 - i as f32 * 0.03;
            let class_id = 0.0;
            data.extend_from_slice(&[x1, y1, x2, y2, score, class_id]);
        }
        Ok(RawOutput { data, shape: vec![n, 6] })
    }
}

#[cfg(feature = "triton")]
pub struct TritonBackend {
    model: ModelConfig,
    cfg: BackendConfig,
    client: triton_client::Client,
}

#[cfg(feature = "triton")]
impl TritonBackend {
    pub async fn connect(model: ModelConfig, cfg: BackendConfig) -> Result<Self> {
        let tr = cfg.triton.clone().ok_or_else(|| anyhow::anyhow!("missing triton config"))?;
        let client = triton_client::Client::new(tr.url)?;
        Ok(Self { model, cfg, client })
    }
}

#[cfg(feature = "triton")]
#[async_trait::async_trait]
impl YoloBackend for TritonBackend {
    async fn infer(&self, input: &Tensor) -> Result<RawOutput> {
        let tr = self.cfg.triton.clone().ok_or_else(|| anyhow::anyhow!("missing triton config"))?;

        let mut infer_input = triton_client::InferInput::new(tr.input_name, input.shape.to_vec(), "FP32")?;
        infer_input.set_data(input.data.clone());

        let outputs = vec![triton_client::InferRequestedOutput::new(tr.output_name)];

        let req = triton_client::InferRequest::new(tr.model_name, tr.model_version.as_deref(), vec![infer_input], outputs);
        let resp = self.client.infer(req).await?;

        let out = resp.get_output(&tr.output_name).ok_or_else(|| anyhow::anyhow!("missing output"))?;
        let data = out.as_f32()?.to_vec();
        let shape = out.shape().to_vec();

        Ok(RawOutput { data, shape })
    }
}
