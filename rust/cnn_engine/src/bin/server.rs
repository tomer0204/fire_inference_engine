// run dummy - cargo run --bin server -- ../yolo.yaml # 
// run Triton - cargo run --features triton --bin server -- ../yolo.yaml


use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::Mutex;

use yolo_engine::io::{EngineConfig, Frame};
use yolo_engine::process::backend::{BackendKind, YoloBackend};
use yolo_engine::{postprocess, preprocess};

#[derive(Clone)]
struct AppState {
    cfg: EngineConfig,
    backend: Arc<Box<dyn YoloBackend>>,
}

#[derive(serde::Deserialize)]
struct InferQuery {
    frame_id: Option<String>,
}

async fn infer_handler(
    State(state): State<Arc<AppState>>,
    bytes: Bytes,
) -> impl IntoResponse {
    let img = match image::load_from_memory(&bytes) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("invalid image: {e}")).into_response(),
    };

    let (width, height) = img.dimensions();
    let frame = Frame { image: img, width, height };

    let input = match preprocess::run(&frame, &state.cfg.preprocess) {
        Ok(v) => v,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("preprocess error: {e}")).into_response(),
    };

    let raw = match state.backend.infer(&input).await {
        Ok(v) => v,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("infer error: {e}")).into_response(),
    };

    let mut dets = match postprocess::run(&raw, &state.cfg.postprocess) {
        Ok(v) => v,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("postprocess error: {e}")).into_response(),
    };

    dets.meta.input_width = frame.width;
    dets.meta.input_height = frame.height;
    dets.meta.model_width = state.cfg.model.input_width;
    dets.meta.model_height = state.cfg.model.input_height;

    (StatusCode::OK, Json(dets)).into_response()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
                anyhow::bail!("Triton feature not enabled. Build with: cargo run --features triton --bin server -- <config>")
            }
        }
    };

    let state = Arc::new(AppState {
        cfg,
        backend: Arc::new(backend),
    });

    let app = Router::new()
        .route("/infer", post(infer_handler))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8088));
    println!("Rust inference server listening on http://{addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}
