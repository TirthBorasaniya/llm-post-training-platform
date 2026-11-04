"""Prometheus text format parsing and metric snapshots from a running server."""

import re
from collections.abc import Iterable

import httpx

# ============= Constants =============

SAMPLE_PATTERN = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)
SCRAPE_TIMEOUT_SECONDS = 10.0
SERVING_METRICS = (
    "sglang:num_queue_reqs",
    "sglang:num_running_reqs",
    "sglang:token_usage",
    "sglang:gen_throughput",
    "sglang:cache_hit_rate",
    "sglang:spec_accept_length",
    "sglang:spec_accept_rate",
)
LLAMACPP_METRICS = ("llamacpp:requests_deferred", "llamacpp:requests_processing")

Sample = tuple[dict[str, str], float]


# ============= Parsing =============


def parse_labels(text: str | None) -> dict[str, str]:
    """Parse ``a="1",b="2"`` into a dict."""
    if not text:
        return {}
    return {key: value for key, value in re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', text)}


def parse_prometheus_text(text: str) -> dict[str, list[Sample]]:
    """
    Parse Prometheus exposition text into samples per metric name.

    Comment lines are skipped; ``NaN`` and ``Inf`` values are kept as floats.
    """
    metric_dict: dict[str, list[Sample]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_PATTERN.match(line)
        if match is None:
            continue
        value = float(match.group("value").replace("+Inf", "inf").replace("-Inf", "-inf"))
        metric_dict.setdefault(match.group("name"), []).append(
            (parse_labels(match.group("labels")), value)
        )
    return metric_dict


def metric_total(metric_dict: dict[str, list[Sample]], name: str) -> float | None:
    """Sum every sample of a metric, or return ``None`` when it is absent."""
    samples = metric_dict.get(name)
    if not samples:
        return None
    return sum(value for _, value in samples)


def metric_names(metric_dict: dict[str, list[Sample]], prefix: str) -> list[str]:
    """List metric names with a prefix, sorted."""
    return sorted(name for name in metric_dict if name.startswith(prefix))


# ============= Scraping =============


def scrape(url: str, client: httpx.Client | None = None) -> dict[str, list[Sample]]:
    """Fetch and parse a metrics endpoint."""
    http = client or httpx.Client(timeout=SCRAPE_TIMEOUT_SECONDS)
    response = http.get(url)
    response.raise_for_status()
    return parse_prometheus_text(response.text)


def snapshot(
    url: str, names: Iterable[str] = SERVING_METRICS, client: httpx.Client | None = None
) -> dict[str, float | None]:
    """Return the totals of selected metrics from one scrape; absent metrics map to ``None``."""
    metric_dict = scrape(url, client)
    return {name: metric_total(metric_dict, name) for name in names}
