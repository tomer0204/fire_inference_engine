# ===============================
# Stage 1 - Builder
# ===============================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

COPY . .
RUN cargo build --release


# ===============================
# Stage 2 - Runtime (USE DEVEL)
# ===============================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libgomp1 \
    ca-certificates \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-gpu-1.18.0.tgz \
    && tar -xzf onnxruntime-linux-x64-gpu-1.18.0.tgz \
    && mv onnxruntime-linux-x64-gpu-1.18.0 /opt/onnxruntime \
    && rm onnxruntime-linux-x64-gpu-1.18.0.tgz

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda/lib64

COPY --from=builder /app/target/release/cnn_engine /app/server

EXPOSE 8088

CMD ["/app/server"]
