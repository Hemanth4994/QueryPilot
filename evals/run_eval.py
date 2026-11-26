"""Golden 30-query Text-to-SQL evaluation harness (offline and live modes)."""

import argparse
from datetime import UTC, date, datetime
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings  # noqa: E402
from backend.app.core import llm_client  # noqa: E402
from backend.app.core.agent import run_query_pipeline  # noqa: E402
from backend.app.core.database import (  # noqa: E402
    CORE_TABLES,
    check_and_seed_db,
    configure_database,
    is_postgres,
)
from backend.app.core.schema_rag import (  # noqa: E402
    index_all_tables,
    retrieve_relevant_tables,
)
from backend.app.core.sql_guard import (  # noqa: E402
    execute_sql_bounded,
    validate_read_only_sql,
)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def normalize_cell(value: Any, float_precision: int = 2) -> Any:
    """Normalize SQL cell values for dialect-independent result-set comparison."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        rounded = round(value, float_precision)
        if rounded == 0:
            return 0
        if rounded.is_integer():
            return int(rounded)
        return rounded
    if isinstance(value, int):
        return value
    if isinstance(value, (pd.Timestamp, datetime, date)):
        iso = value.isoformat()
        return iso[:10] if iso.endswith("T00:00:00") else iso
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith(" 00:00:00") and len(stripped) == 19:
            return stripped[:10]
        return stripped
    if hasattr(value, "item"):
        return normalize_cell(value.item(), float_precision=float_precision)
    return str(value)


def _row_sort_key(row: tuple[Any, ...]) -> tuple[tuple[int, str], ...]:
    """Build a total-order sort key across mixed None/numeric/str cells."""
    return tuple((0, "") if cell is None else (1, repr(cell)) for cell in row)


def normalize_rows(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    order_matters: bool = False,
    float_precision: int = 2,
) -> list[tuple[Any, ...]]:
    """Convert row dicts into normalized tuples, sorting canonically when order is open."""
    if not rows:
        return []
    col_order = columns or list(rows[0].keys())
    normalized = [
        tuple(normalize_cell(row.get(col), float_precision=float_precision) for col in col_order)
        for row in rows
    ]
    if not order_matters:
        normalized.sort(key=_row_sort_key)
    return normalized


def compare_result_sets(
    actual_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    order_matters: bool = False,
    float_precision: int = 2,
) -> tuple[bool, str]:
    """Compare actual vs expected result sets regardless of SQL syntax or alias names."""
    if len(actual_rows) != len(expected_rows):
        return False, f"Row count mismatch: actual={len(actual_rows)} expected={len(expected_rows)}"

    if not expected_rows:
        return True, "Both result sets empty"

    actual_cols = list(actual_rows[0].keys())
    expected_cols = list(expected_rows[0].keys())
    if len(actual_cols) != len(expected_cols):
        return False, f"Column count mismatch: actual={len(actual_cols)} expected={len(expected_cols)}"

    # Align by normalized lowercase column names when identical; otherwise positional tuples
    actual_lower = {c.lower(): c for c in actual_cols}
    expected_lower = [c.lower() for c in expected_cols]
    if set(actual_lower) == set(expected_lower):
        aligned_actual_cols = [actual_lower[c] for c in expected_lower]
    else:
        aligned_actual_cols = actual_cols

    actual_norm = normalize_rows(
        actual_rows,
        columns=aligned_actual_cols,
        order_matters=order_matters,
        float_precision=float_precision,
    )
    expected_norm = normalize_rows(
        expected_rows,
        columns=expected_cols,
        order_matters=order_matters,
        float_precision=float_precision,
    )

    if actual_norm == expected_norm:
        return True, "Matched"
    return False, f"Values mismatch: actual[0]={actual_norm[0]} expected[0]={expected_norm[0]}"


def percentile(values: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile (0-100) from latency samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 2)


def _select_sql_for_dialect(case: dict[str, Any]) -> str:
    """Pick PostgreSQL golden_sql or sqlite_sql fallback when running on SQLite."""
    if not is_postgres() and case.get("sqlite_sql"):
        return str(case["sqlite_sql"])
    return str(case["golden_sql"])


def _offline_mock_embed(text_input: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Deterministic local embedding vector used only in --mode offline."""
    _ = task_type
    lowered = text_input.lower()
    if "create table customers" in lowered:
        return [1.0, 0.0, 0.0, 0.0]
    if "create table products" in lowered:
        return [0.0, 1.0, 0.0, 0.0]
    if "create table orders" in lowered:
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


def run_offline_eval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate golden SQL execution, SQL guard blocks, and INSUFFICIENT_SCHEMA offline."""
    from backend.app.core import schema_rag

    original_embed = schema_rag.generate_embedding
    schema_rag.generate_embedding = _offline_mock_embed  # type: ignore[assignment]
    results: list[dict[str, Any]] = []

    try:
        index_all_tables(force=True, user_id=None)

        for case in cases:
            expected_status = case["expected_status"]
            if expected_status == "SUCCESS":
                sql = _select_sql_for_dialect(case)
                guard_err = validate_read_only_sql(sql, user_id=1, allowed_tables=set(CORE_TABLES))
                if guard_err is not None:
                    results.append(
                        {"id": case["id"], "category": case["category"], "passed": False, "detail": guard_err}
                    )
                    continue
                df, exec_err, _ = execute_sql_bounded(sql, user_id=1, allowed_tables=set(CORE_TABLES))
                passed = exec_err is None and df is not None and not df.empty
                detail = f"rows={len(df)}" if passed else f"Execution failed: {exec_err}"
                results.append(
                    {"id": case["id"], "category": case["category"], "passed": passed, "detail": detail}
                )

            elif expected_status == "INSUFFICIENT_SCHEMA":
                retrieved = retrieve_relevant_tables(case["question"], user_id=1)
                passed = len(retrieved) == 0
                detail = "Short-circuited (0 tables matched)" if passed else f"Unexpected match: {retrieved}"
                results.append(
                    {"id": case["id"], "category": case["category"], "passed": passed, "detail": detail}
                )

            elif expected_status == "BLOCKED":
                guard_err = validate_read_only_sql(
                    case["question"],
                    user_id=1,
                    allowed_tables=set(CORE_TABLES),
                )
                passed = guard_err is not None and "Security Error" in guard_err
                detail = guard_err or "Guard failed to reject adversarial input"
                results.append(
                    {"id": case["id"], "category": case["category"], "passed": passed, "detail": detail}
                )
    finally:
        schema_rag.generate_embedding = original_embed  # type: ignore[assignment]

    return _summarize_report(mode="offline", results=results)


def run_live_eval(cases: list[dict[str, Any]], delay_seconds: float = 0.0) -> dict[str, Any]:
    """Run all 30 prompts live against Gemini + Schema RAG and score Execution Accuracy."""
    llm_client.reset_genai_client()
    index_all_tables(force=False, user_id=None)

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    self_corrections = 0
    fallback_count = 0
    guard_rejections = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for idx, case in enumerate(cases):
        if idx > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        expected_status = case["expected_status"]
        try:
            resp = run_query_pipeline(case["question"], user_id=1, emit_telemetry=True)
        except Exception as exc:
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "passed": False,
                    "detail": f"Pipeline exception: {exc}",
                    "latency_ms": 0.0,
                    "corrected": False,
                    "model": None,
                }
            )
            continue

        latencies.append(resp.latency_ms)
        if resp.corrected:
            self_corrections += 1
        if resp.fallback_triggered:
            fallback_count += 1
        if resp.sql_guard_status == "blocked":
            guard_rejections += 1
        total_input_tokens += resp.input_tokens
        total_output_tokens += resp.output_tokens

        if expected_status == "SUCCESS":
            if resp.status != "SUCCESS":
                passed = False
                detail = f"Expected SUCCESS, got {resp.status}: {resp.error or resp.message}"
            else:
                expected_sql = _select_sql_for_dialect(case)
                exp_df, exp_err, _ = execute_sql_bounded(
                    expected_sql,
                    user_id=1,
                    allowed_tables=set(CORE_TABLES),
                )
                if exp_err or exp_df is None:
                    passed = False
                    detail = f"Golden SQL error: {exp_err}"
                else:
                    expected_rows = exp_df.to_dict(orient="records")
                    passed, detail = compare_result_sets(
                        resp.rows,
                        expected_rows,
                        order_matters=bool(case.get("order_matters", False)),
                    )
        elif expected_status == "INSUFFICIENT_SCHEMA":
            passed = resp.status == "INSUFFICIENT_SCHEMA"
            detail = f"status={resp.status}"
        elif expected_status == "BLOCKED":
            # Either SQL guard rejected unsafe SQL or Schema RAG blocked unknown cross-tenant table
            passed = resp.sql_guard_status == "blocked" or resp.status == "INSUFFICIENT_SCHEMA"
            detail = f"status={resp.status}, guard={resp.sql_guard_status}"
        else:
            passed = False
            detail = f"Unknown expected_status: {expected_status}"

        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "passed": passed,
                "detail": detail,
                "latency_ms": resp.latency_ms,
                "corrected": resp.corrected,
                "model": resp.model,
            }
        )

    total = len(results)
    return _summarize_report(
        mode="live",
        results=results,
        extra_metrics={
            "primary_model": settings.primary_model,
            "fallback_model": settings.fallback_model,
            "embedding_model": settings.embedding_model,
            "database_dialect": "postgresql" if is_postgres() else "sqlite",
            "self_correction_rate": round(self_corrections / total, 4) if total else 0.0,
            "self_corrections": self_corrections,
            "fallback_count": fallback_count,
            "guard_rejections": guard_rejections,
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
    )


def _summarize_report(
    mode: str,
    results: list[dict[str, Any]],
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate pass/fail metrics and per-category counts."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    by_category: dict[str, dict[str, int]] = {}
    for row in results:
        cat = row["category"]
        bucket = by_category.setdefault(cat, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if row["passed"]:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

    report: dict[str, Any] = {
        "mode": mode,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "execution_accuracy": round(passed / total, 4) if total else 0.0,
        "by_category": by_category,
        "results": results,
    }
    if extra_metrics:
        report.update(extra_metrics)
    return report


def write_reports(report: dict[str, Any], output_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    """Write timestamped JSON and Markdown summary reports under evals/reports/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"eval_{report['mode']}_{stamp}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# QueryPilot Evaluation Report ({report['mode'].upper()})",
        "",
        f"- **Timestamp (UTC):** {report['timestamp_utc']}",
        f"- **Passed:** {report['passed']} / {report['total_cases']} ({report['execution_accuracy'] * 100:.2f}%)",
    ]
    if report["mode"] == "live":
        lines.extend(
            [
                f"- **Models (Primary / Fallback / Embed):** {report.get('primary_model')} / {report.get('fallback_model')} / {report.get('embedding_model')}",
                f"- **Database Dialect:** {report.get('database_dialect')}",
                f"- **Self-Correction Rate:** {report['self_correction_rate'] * 100:.2f}% ({report['self_corrections']})",
                f"- **Fallback Count:** {report['fallback_count']}",
                f"- **Guard Rejections:** {report['guard_rejections']}",
                f"- **p50 / p95 Latency (ms):** {report['p50_latency_ms']} / {report['p95_latency_ms']}",
                f"- **Tokens (in / out):** {report['total_input_tokens']} / {report['total_output_tokens']}",
            ]
        )

    lines.extend(["", "## Category Breakdown", "", "| Category | Passed | Total |", "|---|---:|---:|"])
    for cat, counts in report["by_category"].items():
        lines.append(f"| {cat} | {counts['passed']} | {counts['total']} |")

    failures = [item for item in report["results"] if not item["passed"]]
    if failures:
        lines.extend(["", "## Failures", ""])
        for item in failures:
            lines.append(f"- **{item['id']}** ({item['category']}): {item['detail']}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _setup_database(mode: str, database_url: str | None) -> tempfile.TemporaryDirectory | None:
    """Configure target database (PostgreSQL when configured, or temp SQLite offline)."""
    if database_url:
        settings.database_url = database_url
        configure_database(database_url)
        check_and_seed_db()
        return None

    if "DATABASE_URL" in os.environ or "DB_URL" in os.environ:
        configure_database(settings.database_url)
        check_and_seed_db()
        return None

    if mode == "offline":
        temp_dir = tempfile.TemporaryDirectory(prefix="querypilot_eval_")
        db_file = Path(temp_dir.name) / "eval_offline.sqlite3"
        sqlite_url = f"sqlite:///{db_file.as_posix()}"
        settings.database_url = sqlite_url
        configure_database(sqlite_url)
        check_and_seed_db()
        return temp_dir

    configure_database(settings.database_url)
    check_and_seed_db()
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for QueryPilot 30-query golden evaluation."""
    parser = argparse.ArgumentParser(description="Run QueryPilot 30-query golden SQL evaluation.")
    parser.add_argument(
        "--mode",
        choices=("offline", "live"),
        default="offline",
        help="offline validates golden SQL + guards without API calls; live queries Gemini end-to-end.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to golden dataset JSON file.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Optional explicit database connection URL override.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between live cases to respect free-tier RPM limits.",
    )
    args = parser.parse_args(argv)

    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    temp_dir = _setup_database(args.mode, args.database_url)

    try:
        if args.mode == "offline":
            report = run_offline_eval(cases)
        else:
            report = run_live_eval(cases, delay_seconds=args.delay_seconds)
        json_path, md_path = write_reports(report)
        print(
            f"[{report['mode'].upper()}] {report['passed']}/{report['total_cases']} passed "
            f"({report['execution_accuracy'] * 100:.2f}%) -> {md_path}"
        )
        _ = json_path
        return 0 if report["failed"] == 0 else 1
    finally:
        if temp_dir is not None:
            configure_database("sqlite:///:memory:")
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
