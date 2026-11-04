"""Start an inference server, wait for readiness, run a command against it, and stop it."""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import httpx

from llmplatform.serve.engines import (
    DisaggregationSpec,
    ServerSpec,
    SpeculativeSpec,
    build_command,
    build_env,
)

logger = logging.getLogger(__name__)

# ============= Constants =============

DEFAULT_READY_TIMEOUT_SECONDS = 1800.0
POLL_INTERVAL_SECONDS = 3.0
TERMINATE_GRACE_SECONDS = 30.0
PLACEHOLDER_BASE_URL = "{base_url}"
PLACEHOLDER_OPENAI_URL = "{openai_url}"
PLACEHOLDER_METRICS_URL = "{metrics_url}"


# ============= Readiness =============


class ServerStartError(RuntimeError):
    """Raised when a server exits early or never becomes healthy."""


def wait_for_health(
    url: str,
    timeout: float,
    process: subprocess.Popen[Any] | None = None,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    """
    Poll ``url`` until it answers 200.

    Parameters
    ----------
    url : str
        Health endpoint.
    timeout : float
        Seconds to wait before giving up.
    process : Popen, optional
        Server process; its early exit aborts the wait immediately.
    """
    http = client or httpx.Client(timeout=5.0)
    deadline = clock() + timeout
    while clock() < deadline:
        if process is not None and process.poll() is not None:
            raise ServerStartError(f"server exited with status {process.returncode} before ready")
        try:
            if http.get(url).status_code == 200:
                return
        except httpx.TransportError:
            pass
        sleep(interval)
    raise ServerStartError(f"server not healthy after {timeout:.0f} s at {url}")


# ============= Process =============


class ManagedProcess:
    """Context manager owning one background process, its log file, and its readiness probe."""

    def __init__(
        self,
        command: list[str],
        health_url: str,
        log_path: str | Path,
        env: dict[str, str] | None = None,
        ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    ) -> None:
        self.command = command
        self.health_url = health_url
        self.log_path = Path(log_path)
        self.env = env or {}
        self.ready_timeout = ready_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: BinaryIO | None = None

    def start(self) -> None:
        """Launch the process in its own session and wait until it is healthy."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("starting process", extra={"command": " ".join(self.command)})
        log_handle = open(self.log_path, "ab")  # noqa: SIM115 - closed in stop()
        self._log_handle = log_handle
        self.process = subprocess.Popen(
            self.command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, **self.env},
            start_new_session=True,
        )
        try:
            wait_for_health(self.health_url, self.ready_timeout, self.process)
        except ServerStartError:
            self.stop()
            raise

    def stop(self) -> None:
        """Terminate the process group, escalating to SIGKILL after a grace period."""
        process = self.process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self.process = None

    def __enter__(self) -> "ManagedProcess":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


class ManagedServer(ManagedProcess):
    """A ``ManagedProcess`` that runs an inference server described by a ``ServerSpec``."""

    def __init__(
        self,
        spec: ServerSpec,
        log_path: str | Path,
        ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        python: str | None = None,
    ) -> None:
        super().__init__(
            build_command(spec, python), spec.health_url(), log_path, build_env(spec), ready_timeout
        )
        self.spec = spec


# ============= CLI =============


def substitute(arguments: list[str], spec: ServerSpec) -> list[str]:
    """Replace URL placeholders in a command with the running server's addresses."""
    mapping = {
        PLACEHOLDER_BASE_URL: spec.base_url(),
        PLACEHOLDER_OPENAI_URL: spec.openai_url(),
        PLACEHOLDER_METRICS_URL: spec.metrics_url(),
    }
    result = []
    for argument in arguments:
        for token, value in mapping.items():
            argument = argument.replace(token, value)
        result.append(argument)
    return result


def spec_from_args(args: argparse.Namespace) -> ServerSpec:
    """Build a ``ServerSpec`` from parsed command-line options."""
    speculative = None
    if args.draft_model:
        speculative = SpeculativeSpec(
            draft_model_path=args.draft_model,
            num_steps=args.spec_steps,
            eagle_topk=args.spec_topk,
            num_draft_tokens=args.spec_draft_tokens,
        )
    disaggregation = None
    if args.disaggregation_mode:
        disaggregation = DisaggregationSpec(
            mode=args.disaggregation_mode,
            transfer_backend=args.transfer_backend,
            bootstrap_port=args.bootstrap_port,
        )
    binary = None
    if args.llama_cpp_dir:
        from llmplatform.quantize.gguf import resolve_tool

        binary = resolve_tool(args.llama_cpp_dir, "serve")
    return ServerSpec(
        engine=args.engine,
        model_path=args.model,
        port=args.port,
        served_model_name=args.served_model_name,
        quantization=args.quantization,
        prefix_cache=not args.no_prefix_cache,
        max_model_len=args.max_model_len,
        mem_fraction=args.mem_fraction,
        gpu_ids=[int(item) for item in args.gpu_ids.split(",")] if args.gpu_ids else None,
        extra_args=args.server_arg,
        speculative=speculative,
        disaggregation=disaggregation,
        binary=binary,
        env_extra=dict(item.split("=", 1) for item in args.env),
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for ``python -m llmplatform.serve.launch``."""
    parser = argparse.ArgumentParser(
        description="Run a command against a managed inference server. Put the command after --; "
        "{base_url}, {openai_url}, and {metrics_url} are replaced with the server addresses."
    )
    parser.add_argument("--engine", choices=["sglang", "vllm", "llamacpp"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--mem-fraction", type=float, default=None)
    parser.add_argument("--gpu-ids", default=None, help="comma-separated GPU indices")
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--spec-steps", type=int, default=2)
    parser.add_argument("--spec-topk", type=int, default=1)
    parser.add_argument("--spec-draft-tokens", type=int, default=3)
    parser.add_argument("--disaggregation-mode", choices=["prefill", "decode"], default=None)
    parser.add_argument("--transfer-backend", default="nixl")
    parser.add_argument("--bootstrap-port", type=int, default=None)
    parser.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the server")
    parser.add_argument("--llama-cpp-dir", default=None, help="built llama.cpp checkout")
    parser.add_argument("--log", default="logs/server.log")
    parser.add_argument("--ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT_SECONDS)
    parser.add_argument(
        "--print-command", action="store_true", help="print the server command only"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Start the server, run the trailing command, stop the server, return the command's status."""
    args = build_parser().parse_args(argv)
    spec = spec_from_args(args)
    if args.print_command:
        print(" ".join(build_command(spec)))
        return 0
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise SystemExit("a command to run after -- is required")
    with ManagedServer(spec, args.log, args.ready_timeout):
        completed = subprocess.run(substitute(command, spec), check=False)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
