"""Tests for the Helm chart, cluster assets, and the autoscaling report."""

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from llmplatform.bench.k8s_report import (
    build_report,
    find_breach,
    find_ready,
    latency_stats,
    parse_timestamp,
    read_requests,
    read_series,
)

pytestmark = pytest.mark.unit

CPU_SET = [
    "--set",
    "model.repo=example/model-GGUF",
    "--set",
    "model.file=model.gguf",
    "--set",
    f"model.sha256={'0' * 64}",
]
needs_helm = pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")


def helm(root: Path, tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HELM_CACHE_HOME": str(tmp_path / "cache"),
        "HELM_CONFIG_HOME": str(tmp_path / "config"),
        "HELM_DATA_HOME": str(tmp_path / "data"),
    }
    return subprocess.run(["helm", *args], capture_output=True, text=True, cwd=root, env=env)


def render(root: Path, tmp_path: Path, values: str, *extra: str) -> list[dict]:
    result = helm(
        root,
        tmp_path,
        "template",
        "r",
        "charts/llm-serve",
        "--values",
        f"charts/llm-serve/{values}",
        *extra,
    )
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


# ============= Chart =============


@needs_helm
def test_cpu_chart_renders_llamacpp_with_checks(root: Path, tmp_path: Path) -> None:
    docs = render(
        root,
        tmp_path,
        "values-cpu-kind.yaml",
        *CPU_SET,
        "--set",
        "autoscaling.enabled=true",
        "--set",
        "serviceMonitor.enabled=true",
    )
    kinds = {doc["kind"] for doc in docs}
    assert kinds == {
        "Deployment",
        "Service",
        "PodDisruptionBudget",
        "ScaledObject",
        "ServiceMonitor",
    }
    deployment = next(doc for doc in docs if doc["kind"] == "Deployment")
    spec = deployment["spec"]["template"]["spec"]
    container = spec["containers"][0]
    assert ":latest" not in container["image"] and "server-b" in container["image"]
    assert (
        container["args"][:3] == ["-m", "/models/model.gguf", "--host"]
        and "--metrics" in container["args"]
    )
    assert {"startupProbe", "readinessProbe", "livenessProbe"} <= set(container)
    assert container["startupProbe"]["failureThreshold"] >= 60
    assert spec["initContainers"][0]["name"] == "fetch-model"
    pdb = next(doc for doc in docs if doc["kind"] == "PodDisruptionBudget")
    assert pdb["spec"]["minAvailable"] == 1
    service = next(doc for doc in docs if doc["kind"] == "Service")
    assert {port["name"] for port in service["spec"]["ports"]} == {"http", "metrics"}
    scaled = next(doc for doc in docs if doc["kind"] == "ScaledObject")["spec"]
    assert (scaled["minReplicaCount"], scaled["maxReplicaCount"], scaled["cooldownPeriod"]) == (
        1,
        3,
        120,
    )
    trigger = scaled["triggers"][0]
    assert trigger["type"] == "prometheus"
    assert "llamacpp:requests_deferred" in trigger["metadata"]["query"]
    assert str(trigger["metadata"]["threshold"]) == "2"


@needs_helm
def test_gpu_chart_renders_sglang(root: Path, tmp_path: Path) -> None:
    docs = render(
        root,
        tmp_path,
        "values-gpu-sglang.yaml",
        "--set",
        "model.repo=example/model",
        "--set",
        "autoscaling.enabled=true",
    )
    deployment = next(doc for doc in docs if doc["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert "--enable-metrics" in container["args"]
    assert container["command"] == ["python3", "-m", "sglang.launch_server"]
    scaled = next(doc for doc in docs if doc["kind"] == "ScaledObject")["spec"]
    assert "sglang:num_queue_reqs" in scaled["triggers"][0]["metadata"]["query"]


@needs_helm
def test_chart_rejects_bad_input(root: Path, tmp_path: Path) -> None:
    base = [
        "template",
        "r",
        "charts/llm-serve",
        "--values",
        "charts/llm-serve/values-cpu-kind.yaml",
    ]
    bad_backend = helm(root, tmp_path, *base, *CPU_SET, "--set", "backend=other")
    assert bad_backend.returncode != 0 and "backend" in bad_backend.stderr
    no_hash = helm(
        root, tmp_path, *base, "--set", "model.repo=example/m", "--set", "model.file=m.gguf"
    )
    assert no_hash.returncode != 0 and "sha256" in no_hash.stderr
    malformed = helm(
        root,
        tmp_path,
        *base,
        "--set",
        "model.repo=example/m",
        "--set",
        "model.file=m.gguf",
        "--set",
        "model.sha256=abc",
    )
    assert malformed.returncode != 0


@needs_helm
def test_chart_lints_for_both_backends(root: Path, tmp_path: Path) -> None:
    cpu = helm(
        root,
        tmp_path,
        "lint",
        "charts/llm-serve",
        "--values",
        "charts/llm-serve/values-cpu-kind.yaml",
        *CPU_SET,
    )
    gpu = helm(
        root,
        tmp_path,
        "lint",
        "charts/llm-serve",
        "--values",
        "charts/llm-serve/values-gpu-sglang.yaml",
        "--set",
        "model.repo=example/model",
    )
    assert cpu.returncode == 0, cpu.stdout + cpu.stderr
    assert gpu.returncode == 0, gpu.stdout + gpu.stderr


def test_chart_defaults_pin_images_and_embed_no_model_identity(root: Path) -> None:
    values = yaml.safe_load((root / "charts/llm-serve/values.yaml").read_text())
    for image in (
        values["images"]["llamacpp"],
        values["images"]["sglang"],
        values["images"]["init"],
    ):
        assert image["tag"] and image["tag"] != "latest"
    assert values["model"]["repo"] == "" and values["model"]["sha256"] == ""
    assert values["autoscaling"]["threshold"] == 2 and values["pdb"]["minAvailable"] == 1


# ============= Cluster assets =============


def test_cluster_assets_parse(root: Path) -> None:
    kind = yaml.safe_load((root / "k8s/kind-config.yaml").read_text())
    roles = [node["role"] for node in kind["nodes"]]
    assert roles.count("control-plane") == 1 and roles.count("worker") == 2
    json.loads((root / "k8s/dashboards/llm-serve-dashboard.json").read_text())
    for name in ("monitoring-values.yaml", "keda-values.yaml"):
        assert isinstance(yaml.safe_load((root / "k8s" / name).read_text()), dict)
    script = (root / "k6/chat_ramp.js").read_text()
    assert "ramping-vus" in script and "/v1/chat/completions" in script


@pytest.mark.parametrize("name", ["k8s_up.sh", "k8s_load.sh", "k8s_down.sh", "validate_chart.sh"])
def test_k8s_scripts_have_valid_syntax(root: Path, name: str) -> None:
    result = subprocess.run(
        ["bash", "-n", str(root / "scripts" / name)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["k8s_up.sh", "k8s_load.sh", "k8s_down.sh"])
def test_k8s_scripts_dry_run_never_touches_docker_or_kind(root: Path, name: str) -> None:
    env = {"PATH": "/usr/bin:/bin", "DRY_RUN": "1", "HOME": str(root / ".tmp")}
    result = subprocess.run(
        [str(root / "scripts" / name)], capture_output=True, text=True, env=env, cwd=root
    )
    assert result.returncode == 0, result.stderr
    assert "+" in result.stdout


# ============= Report =============


def series_row(second: int, replicas: int, ready: int, depth: float | None) -> dict:
    base = parse_timestamp("2026-01-01T00:00:00Z")
    stamp = datetime.fromtimestamp(base + second, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "ts": stamp,
        "t": base + second,
        "replicas": replicas,
        "ready_replicas": ready,
        "queue_depth": depth,
    }


def request_row(second: float, duration_ms: float, status: int = 200, tokens: int = 10) -> dict:
    base = parse_timestamp("2026-01-01T00:00:00Z")
    return {"end_s": base + second, "duration_ms": duration_ms, "status": status, "tokens": tokens}


def test_timestamp_and_latency_stats() -> None:
    assert parse_timestamp("1970-01-01T00:01:00Z") == 60.0
    assert latency_stats([]) == {"count": 0, "p50_ms": None, "p95_ms": None}
    stats = latency_stats([100.0] * 19 + [900.0])
    assert stats["p50_ms"] == 100.0 and stats["p95_ms"] == 900.0


def test_breach_and_ready_detection() -> None:
    series = [
        series_row(0, 1, 1, 0.0),
        series_row(5, 1, 1, None),
        series_row(10, 1, 1, 3.0),
        series_row(15, 2, 1, 4.0),
        series_row(40, 2, 2, 1.0),
    ]
    breach = find_breach(series, 2.0)
    assert breach is not None and breach["ts"].endswith(":10Z")
    ready = find_ready(series, breach)
    assert ready is not None and ready["ts"].endswith(":40Z")
    assert find_breach(series[:2], 2.0) is None


def test_report_for_a_scale_out_run() -> None:
    series = [
        series_row(0, 1, 1, 0.0),
        series_row(10, 1, 1, 3.0),
        series_row(20, 2, 1, 5.0),
        series_row(40, 2, 2, 1.0),
        series_row(60, 2, 2, 0.0),
    ]
    requests = [request_row(s, 4000.0) for s in (2, 5, 8)] + [
        request_row(s, 1500.0) for s in (45, 50, 55)
    ]
    requests.append(request_row(30, 0.0, status=500, tokens=0))
    report = build_report(series, requests, window_seconds=20.0)
    assert report["scale_out_observed"] is True
    assert report["seconds_from_breach_to_pod_ready"] == 30.0
    assert report["seconds_to_threshold_breach"] == 10.0
    assert report["peak_replicas"] == 2 and report["peak_ready_replicas"] == 2
    assert report["latency_before_scale_out"]["p50_ms"] == 4000.0
    assert report["latency_after_scale_out"]["p50_ms"] == 1500.0
    assert report["requests"] == {"total": 7, "ok": 6, "failed": 1}
    assert report["tokens_per_second"] > 0
    assert report["queue_depth_over_time"][1] == {"seconds": 10.0, "queue_depth": 3.0}
    assert report["replicas_over_time"][-1]["ready_replicas"] == 2


def test_report_without_scale_out_has_no_latency_split() -> None:
    series = [series_row(0, 1, 1, 0.0), series_row(10, 1, 1, 1.0)]
    report = build_report(series, [request_row(1, 100.0)])
    assert report["scale_out_observed"] is False
    assert report["seconds_from_breach_to_pod_ready"] is None
    assert report["latency_before_scale_out"] is None
    with pytest.raises(ValueError):
        build_report([], [])


def test_report_readers(tmp_path: Path) -> None:
    series = tmp_path / "s.jsonl"
    series.write_text(
        json.dumps(
            {"ts": "2026-01-01T00:00:10Z", "replicas": 1, "ready_replicas": 1, "queue_depth": 0}
        )
        + "\n"
        + json.dumps(
            {"ts": "2026-01-01T00:00:00Z", "replicas": 1, "ready_replicas": 1, "queue_depth": None}
        )
        + "\n"
    )
    requests = tmp_path / "r.jsonl"
    requests.write_text(
        json.dumps({"end_ms": 2000, "duration_ms": 5, "status": 200, "tokens": 1}) + "\n"
    )
    assert [row["ts"][-3:-1] for row in read_series(series)] == ["00", "10"]
    assert read_requests(requests)[0]["end_s"] == 2.0
