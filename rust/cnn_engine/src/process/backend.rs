use anyhow::Result;
use async_trait::async_trait;
use ort::{
    environment::Environment,
    session::SessionBuilder,
    tensor::OrtOwnedTensor,
    value::Value,
};
use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
use std::path::Path;
use std::sync::Arc;

use crate::io::{RawOutput, Tensor};

#[async_trait]
pub trait YoloBackend: Send + Sync {
    async fn infer(&mut self, input: &Tensor) -> Result<RawOutput>;
}

pub struct OnnxBackend {
    session: ort::session::Session,
    input_name: String,
}

impl OnnxBackend {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {

        let environment = Arc::new(
            Environment::builder()
                .with_name("fire-engine")
                .build()?,
        );

        let cuda = CUDAExecutionProvider::default();

        let session = SessionBuilder::new(&environment)?
            .with_execution_providers([cuda.into()])?
            .with_model_from_file(path)?;

        let input_name = session.inputs[0].name.clone();

        Ok(Self {
            session,
            input_name,
        })
    }
}

#[async_trait]
impl YoloBackend for OnnxBackend {
    async fn infer(&mut self, input: &Tensor) -> Result<RawOutput> {

        let dims = vec![
            input.shape[0] as i64,
            input.shape[1] as i64,
            input.shape[2] as i64,
            input.shape[3] as i64,
        ];

        let input_tensor = Value::from_array(
            self.session.allocator(),
            &dims,
            &input.data,
        )?;

        let outputs = self.session.run(vec![(
            self.input_name.as_str(),
            input_tensor,
        )])?;

        let output_tensor: OrtOwnedTensor<f32, _> =
            outputs[0].try_extract()?;

        let shape = output_tensor
            .view()
            .shape()
            .iter()
            .map(|d| *d as usize)
            .collect();

        let data = output_tensor.view().iter().cloned().collect();

        Ok(RawOutput { shape, data })
    }
}
