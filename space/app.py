"""Public demo: the distilled student next to the base student, both Q4_K_M GGUF on the CPU."""

import os
import threading
from collections.abc import Callable, Iterator
from typing import Any

import gradio as gr
from limits import MAX_NEW_TOKENS_CAP, SessionRateLimiter
from logic import prepare_request, stream_text

# ============= Configuration =============

GGUF_REPO = os.environ.get("GGUF_REPO", "")
DISTILLED_FILE = os.environ.get("DISTILLED_FILE", "distilled_1_5b-Q4_K_M.gguf")
BASE_FILE = os.environ.get("BASE_FILE", "base_1_5b-Q4_K_M.gguf")
RESULTS_URL = os.environ.get("RESULTS_URL", "")
CONTEXT_TOKENS = 2048
THREADS = max(1, (os.cpu_count() or 2))
TITLE = "Distilled student versus base student"
DESCRIPTION = (
    "Both models are Qwen 2.5 1.5B Instruct in Q4_K_M quantization running on a shared CPU. "
    "The distilled student was trained to match an aligned 7B teacher with on-policy "
    "distillation. Each session may send a few requests per minute and at most "
    f"{MAX_NEW_TOKENS_CAP} new tokens per answer. A free-tier Space sleeps when idle, so the first "
    "request after a pause can take a minute."
)

LIMITER = SessionRateLimiter()
_MODELS: dict[str, Any] = {}
_LOCK = threading.Lock()


# ============= Models =============


def load_model(filename: str) -> Any:
    """Download a GGUF from the Hub on first use and load it with llama.cpp."""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    if not GGUF_REPO:
        raise gr.Error("GGUF_REPO is not configured for this Space")
    path = hf_hub_download(repo_id=GGUF_REPO, filename=filename)
    return Llama(model_path=path, n_ctx=CONTEXT_TOKENS, n_threads=THREADS, verbose=False)


def get_model(filename: str, loader: Callable[[str], Any] = load_model) -> Any:
    """Return a cached model, loading it once under a lock."""
    with _LOCK:
        if filename not in _MODELS:
            _MODELS[filename] = loader(filename)
        return _MODELS[filename]


# ============= Interface =============


def make_compare(
    loader: Callable[[str], Any] = load_model,
) -> Callable[..., Iterator[tuple[str, str]]]:
    """Return the Gradio handler that streams both answers, one model after the other."""

    def compare(
        prompt: str, max_tokens: float, temperature: float, request: gr.Request | None = None
    ) -> Iterator[tuple[str, str]]:
        session = (
            request.session_hash if request is not None and request.session_hash else "anonymous"
        )
        try:
            text, tokens, temp = prepare_request(LIMITER, session, prompt, max_tokens, temperature)
        except (ValueError, PermissionError) as exc:
            raise gr.Error(str(exc)) from exc
        distilled, base = "", ""
        for distilled in stream_text(get_model(DISTILLED_FILE, loader), text, tokens, temp):
            yield distilled, base
        for base in stream_text(get_model(BASE_FILE, loader), text, tokens, temp):
            yield distilled, base

    return compare


def links_markdown() -> str:
    """Return the links shown under the answers."""
    lines = []
    if GGUF_REPO:
        lines.append(f"- Model cards and files: https://huggingface.co/{GGUF_REPO}")
    if RESULTS_URL:
        lines.append(f"- Evaluation results: {RESULTS_URL}")
    return "\n".join(lines) or "Links are configured through the Space variables."


def build_demo(loader: Callable[[str], Any] = load_model) -> gr.Blocks:
    """Build the two-column comparison interface."""
    with gr.Blocks(title=TITLE, analytics_enabled=False) as demo:
        gr.Markdown(f"# {TITLE}\n\n{DESCRIPTION}")
        prompt = gr.Textbox(label="Prompt", lines=4, max_lines=8)
        with gr.Row():
            max_tokens = gr.Slider(
                16, MAX_NEW_TOKENS_CAP, value=128, step=16, label="Max new tokens"
            )
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
        submit = gr.Button("Compare", variant="primary")
        with gr.Row():
            distilled_box = gr.Textbox(label="Distilled student (Q4_K_M)", lines=14)
            base_box = gr.Textbox(label="Base student (Q4_K_M)", lines=14)
        gr.Markdown(links_markdown())
        submit.click(
            make_compare(loader), [prompt, max_tokens, temperature], [distilled_box, base_box]
        )
    return demo


if __name__ == "__main__":
    build_demo().queue(max_size=8).launch(server_name="0.0.0.0", server_port=7860)
