use anyhow::Result;
use yolo_engine::io::{EngineConfig, Frame};
use yolo_engine::process::backend::{BackendKind, YoloBackend};
use yolo_engine::{preprocess, postprocess};

#[tokio::main]
async fn main() -> Result<()> {
    let cfg_path = std::env::args().nth(1).unwrap_or_else(|| "yolo.yaml".to_string());
    let cfg = EngineConfig::from_yaml(&cfg_path)?;

    let backend: Box<dyn YoloBackend> = match cfg.backend.kind {
        BackendKind::Dummy => Box::new(yolo_engine::process::backend::DummyBackend::new(cfg.model.clone())),
        BackendKind::Triton => {
            #[cfg(feature = "triton")]
            {
                Box::new(yolo_engine::process::backend::TritonBackend::connect(cfg.model.clone(), cfg.backend.clone()).await?)
            }
            #[cfg(not(feature = "triton"))]
            {
                anyhow::bail!("Triton feature not enabled. Build with: cargo run --features triton -- <config>")
            }
        }
    };

    let frame = Frame::from_path(&cfg.input.source)?;
    let input = preprocess::run(&frame, &cfg.preprocess)?;
    let raw = backend.infer(&input).await?;
    let detections = postprocess::run(&raw, &cfg.postprocess)?;

    let out = serde_json::to_string_pretty(&detections)?;
    println!("{out}");
    Ok(())
}
