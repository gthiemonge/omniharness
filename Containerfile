# omniharness with llama-server (for model="auto" routing with the d1 model).
#
#   podman build -t omniharness .
#
#   # Once: the d1 decision model (1.7 GB) into a volume, with the image's hf CLI
#   podman volume create agent-hf
#   podman run --rm --userns=keep-id -v agent-hf:/home/agent/.cache/huggingface:Z \
#       --entrypoint hf omniharness download LiquidAI/d1-2B-GGUF d1-2B-Q8_0.gguf
#
#   # Run an agent on the current directory
#   podman run --rm -it --userns=keep-id -e ANTHROPIC_API_KEY \
#       -v agent-hf:/home/agent/.cache/huggingface:Z -v "$PWD:/work:Z" \
#       omniharness --model auto --tools Read,Grep,Glob "What does this project do?"
#
# See the "Container" section of README.md for more (OpenAI, presets, examples).

ARG FEDORA_VERSION=44

# ── llama-server ───────────────────────────────────────────────────────────
FROM registry.fedoraproject.org/fedora:${FEDORA_VERSION} AS llama

# v0.8.0 or later: d1-2B needs the lfm3 architecture and /v1/systemone.
ARG LLAMA_CPP_REF=v0.8.0

RUN dnf install -y --setopt=install_weak_deps=False gcc-c++ cmake ninja-build git-core \
    && dnf clean all

RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggml-org/llama.cpp /src

# Portable build, as llama.cpp's own images: every CPU variant is built and the
# best one for the host is loaded at startup (no -march=native).
# LLAMA_USE_PREBUILT_UI=OFF: no web UI download; the router doesn't need it.
RUN cmake -S /src -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON \
    && cmake --build /build --target llama-server -j "$(nproc)" \
    && mkdir -p /opt/llama \
    && cp -a /build/bin/llama-server /build/bin/*.so* /opt/llama/

# ── omniharness ──────────────────────────────────────────────────────────
FROM registry.fedoraproject.org/fedora:${FEDORA_VERSION}

# libgomp: llama-server's OpenMP runtime. git, ripgrep: for the agents' tools.
RUN dnf install -y --setopt=install_weak_deps=False \
        python3 libgomp git-core ripgrep ca-certificates \
    && dnf clean all

COPY --from=llama /opt/llama /opt/llama

# The python package, both provider SDKs (the Claude Agent SDK bundles its own
# Claude Code binary: no Node.js needed) and the hf CLI to fetch models.
COPY pyproject.toml README.md /opt/omniharness/
COPY src /opt/omniharness/src
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir "/opt/omniharness[all]" huggingface_hub

# Examples, the routing eval and its script, to run from the image.
COPY examples /opt/omniharness/examples
COPY evals /opt/omniharness/evals
COPY scripts /opt/omniharness/scripts

# Agents run as a regular user; mount your project on /work. Mounted projects
# can be owned by another uid: let git use them anyway.
RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /work /home/agent/.cache/huggingface \
    && chown agent:agent /work /home/agent/.cache/huggingface \
    && git config --system --add safe.directory '*'

ENV PATH=/opt/venv/bin:$PATH \
    LLAMA_SERVER=/opt/llama/llama-server \
    LD_LIBRARY_PATH=/opt/llama \
    HF_HOME=/home/agent/.cache/huggingface \
    PYTHONUNBUFFERED=1

USER agent
WORKDIR /work
# The d1 model lives here; mount a volume to keep it across runs.
VOLUME /home/agent/.cache/huggingface

ENTRYPOINT ["omniharness"]
CMD ["--help"]
