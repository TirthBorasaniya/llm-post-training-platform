"""Server stacks for the disaggregated setup and the colocated baseline."""

import sys
from contextlib import ExitStack
from pathlib import Path

from llmplatform.disagg.config import DisaggConfig
from llmplatform.serve.engines import DisaggregationSpec, ServerSpec
from llmplatform.serve.launch import ManagedProcess, ManagedServer

# ============= Constants =============

HOST = "127.0.0.1"
SETUPS = ("disaggregated", "colocated")


# ============= Router =============


def router_command(setup: str, config: DisaggConfig, python: str | None = None) -> list[str]:
    """
    Return the ``sglang_router`` command for a setup.

    The disaggregated router sends each request to the prefill server first and streams from
    the decode server; the colocated router balances two full replicas.
    """
    ports = config.ports
    command = [
        python or sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--host",
        HOST,
        "--port",
        str(ports.router),
        "--policy",
        config.router.policy,
    ]
    prefill_url, decode_url = f"http://{HOST}:{ports.prefill}", f"http://{HOST}:{ports.decode}"
    if setup == "disaggregated":
        return [
            *command,
            "--pd-disaggregation",
            "--prefill",
            prefill_url,
            str(ports.bootstrap),
            "--decode",
            decode_url,
        ]
    if setup == "colocated":
        return [*command, "--worker-urls", prefill_url, decode_url]
    raise ValueError(f"unknown setup {setup!r}; expected one of {SETUPS}")


# ============= Specs =============


def server_specs(setup: str, config: DisaggConfig, model_path: str) -> list[ServerSpec]:
    """
    Return the two server specs of a setup, pinned to GPU 0 and GPU 1.

    Disaggregated: a prefill server on GPU 0 and a decode server on GPU 1. Colocated: two
    identical replicas, one per GPU.
    """
    ports, server = config.ports, config.server
    common = {
        "model_path": model_path,
        "served_model_name": config.model,
        "max_model_len": server.max_model_len,
        "mem_fraction": server.mem_fraction,
    }
    if setup == "disaggregated":
        return [
            ServerSpec(
                "sglang",
                port=ports.prefill,
                gpu_ids=[0],
                disaggregation=DisaggregationSpec(
                    "prefill", config.transfer_backend, ports.bootstrap
                ),
                **common,  # type: ignore[arg-type]
            ),
            ServerSpec(
                "sglang",
                port=ports.decode,
                gpu_ids=[1],
                disaggregation=DisaggregationSpec("decode", config.transfer_backend),
                **common,  # type: ignore[arg-type]
            ),
        ]
    if setup == "colocated":
        return [
            ServerSpec("sglang", port=ports.prefill, gpu_ids=[0], **common),  # type: ignore[arg-type]
            ServerSpec("sglang", port=ports.decode, gpu_ids=[1], **common),  # type: ignore[arg-type]
        ]
    raise ValueError(f"unknown setup {setup!r}; expected one of {SETUPS}")


def start_stack(
    stack: ExitStack, setup: str, config: DisaggConfig, model_path: str, log_dir: str | Path
) -> str:
    """
    Start both servers and the router inside an ``ExitStack`` and return the router URL.

    Everything started is stopped when the stack closes, including after a failure.
    """
    timeout = config.server.ready_timeout_seconds
    for index, spec in enumerate(server_specs(setup, config, model_path)):
        role = spec.disaggregation.mode if spec.disaggregation else f"replica{index}"
        stack.enter_context(ManagedServer(spec, Path(log_dir) / f"{setup}_{role}.log", timeout))
    router_url = f"http://{HOST}:{config.ports.router}"
    stack.enter_context(
        ManagedProcess(
            router_command(setup, config),
            f"{router_url}/health",
            Path(log_dir) / f"{setup}_router.log",
            ready_timeout=timeout,
        )
    )
    return router_url
