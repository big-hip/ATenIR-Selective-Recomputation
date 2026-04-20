"""Tests for toolkit.output (console, charts, export)."""

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from toolkit.output.console import _stringify, _BYTE_KEYS, print_comparison_table, print_step_result
from toolkit.output.export import to_json, to_csv
from toolkit.utils import format_bytes


# ── Sample data ─────────────────────────────────────────────────────

@dataclass
class _FakeResult:
    name: str = "test"
    param_bytes: int = 1048576  # 1 MB
    grad_bytes: int = 1048576
    fw_peak: int = 2097152  # 2 MB
    bw_peak: int = 3145728  # 3 MB
    opt_peak: int = 1572864  # 1.5 MB
    true_peak: int = 3145728
    fwbw_peak: int = 3145728
    base: int = 2097152
    optimizer_bytes: int = 2097152
    elapsed_ms: float = 12.345


SAMPLE_DICTS = [
    {"name": "baseline", "param_bytes": 1048576, "fw_peak": 2097152, "mre": 0.069},
    {"name": "ac", "param_bytes": 1048576, "fw_peak": 1572864, "mre": 0.042},
]


# ── console._stringify tests ────────────────────────────────────────

class TestStringify:
    def test_byte_key_formats(self):
        """Integer values with byte-type keys should be formatted as human-readable."""
        result = _stringify(1048576, key="param_bytes")
        assert "MB" in result or "KB" in result

    def test_non_byte_key_passthrough(self):
        """Integer values with non-byte keys pass through unchanged."""
        result = _stringify(42, key="n_layers")
        assert result == 42

    def test_float_formatting(self):
        """Floats are formatted to 3 decimal places."""
        result = _stringify(0.06912, key="mre")
        assert result == "0.069"

    def test_string_passthrough(self):
        result = _stringify("hello", key="name")
        assert result == "hello"

    def test_all_byte_keys_recognized(self):
        """Spot-check important keys are in _BYTE_KEYS."""
        important = {"param_bytes", "grad_bytes", "fw_peak", "bw_peak", "opt_peak",
                     "true_peak", "fwbw_peak", "estimated_peak", "overall_peak",
                     "l25_true_peak", "l3_true_peak", "sched_fw_peak"}
        assert important.issubset(_BYTE_KEYS)


# ── console.print_comparison_table tests ────────────────────────────

class TestPrintTable:
    def test_prints_without_error(self, capsys):
        print_comparison_table(SAMPLE_DICTS, title="Test")
        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "baseline" in captured.out
        assert "ac" in captured.out

    def test_empty_results(self, capsys):
        print_comparison_table([], title="Empty")
        captured = capsys.readouterr()
        assert "no results" in captured.out

    def test_dataclass_input(self, capsys):
        print_step_result(_FakeResult())
        captured = capsys.readouterr()
        assert "test" in captured.out


# ── export.to_json tests ───────────────────────────────────────────

class TestToJson:
    def test_writes_list(self, tmp_path):
        p = to_json(SAMPLE_DICTS, tmp_path / "out.json")
        data = json.loads(p.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "baseline"

    def test_writes_single_dict(self, tmp_path):
        p = to_json(SAMPLE_DICTS[0], tmp_path / "single.json")
        data = json.loads(p.read_text())
        assert isinstance(data, dict)
        assert data["name"] == "baseline"

    def test_writes_dataclass(self, tmp_path):
        p = to_json([_FakeResult()], tmp_path / "dc.json")
        data = json.loads(p.read_text())
        assert data[0]["name"] == "test"

    def test_creates_parent_dirs(self, tmp_path):
        p = to_json(SAMPLE_DICTS, tmp_path / "sub" / "dir" / "out.json")
        assert p.exists()


# ── export.to_csv tests ────────────────────────────────────────────

class TestToCsv:
    def test_writes_csv(self, tmp_path):
        p = to_csv(SAMPLE_DICTS, tmp_path / "out.csv")
        with p.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["name"] == "baseline"
        assert rows[1]["name"] == "ac"

    def test_fieldname_union(self, tmp_path):
        """CSV fieldnames should be union of all row keys, preserving order."""
        data = [
            {"a": 1, "b": 2},
            {"b": 3, "c": 4},
        ]
        p = to_csv(data, tmp_path / "union.csv")
        with p.open() as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {"a", "b", "c"}

    def test_creates_parent_dirs(self, tmp_path):
        p = to_csv(SAMPLE_DICTS, tmp_path / "nested" / "out.csv")
        assert p.exists()


# ── charts tests (no display, Agg backend) ─────────────────────────

class TestCharts:
    """Verify chart functions execute without error and return figure objects."""

    def test_bar_chart_memory(self):
        from toolkit.output.charts import bar_chart_memory
        fig = bar_chart_memory(SAMPLE_DICTS)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_bar_chart_memory_save(self, tmp_path):
        from toolkit.output.charts import bar_chart_memory
        path = tmp_path / "bar.png"
        bar_chart_memory(SAMPLE_DICTS, save_path=str(path))
        assert path.exists()

    def test_line_chart_mre(self):
        from toolkit.output.charts import line_chart_mre
        fig = line_chart_mre(SAMPLE_DICTS)
        assert fig is not None

    def test_phase_timeline_chart(self):
        from toolkit.output.charts import phase_timeline_chart
        items = [
            {"name": "L2", "base": 100, "fw_peak": 300, "after_fw": 250,
             "bw_peak": 350, "after_bw": 200, "opt_peak": 280, "after_opt": 150},
        ]
        fig = phase_timeline_chart(items)
        assert fig is not None

    def test_phase_grouped_bar(self):
        from toolkit.output.charts import phase_grouped_bar
        items = [
            {"name": "baseline", "fw_peak": 300, "bw_peak": 400, "opt_peak": 200},
            {"name": "ac", "fw_peak": 200, "bw_peak": 250, "opt_peak": 200},
        ]
        fig = phase_grouped_bar(items)
        assert fig is not None

    def test_savings_waterfall(self):
        from toolkit.output.charts import savings_waterfall
        items = [
            {"name": "baseline", "true_peak": 1000},
            {"name": "ac", "true_peak": 700},
        ]
        fig = savings_waterfall(items)
        assert fig is not None

    def test_stacked_breakdown(self):
        from toolkit.output.charts import stacked_breakdown
        items = [
            {"name": "test", "param_bytes": 100, "grad_bytes": 100,
             "optimizer_bytes": 200, "activation_bytes": 500},
        ]
        fig = stacked_breakdown(items)
        assert fig is not None

    def test_stacked_breakdown_l2_fallback(self):
        """L2 results don't have activation_bytes — verify fallback works."""
        from toolkit.output.charts import stacked_breakdown
        items = [
            {"name": "L2", "param_bytes": 1000, "grad_bytes": 1000,
             "optimizer_bytes": 2000, "fwbw_peak": 5000, "base": 3000},
        ]
        fig = stacked_breakdown(items)
        assert fig is not None

    def test_heatmap_strategy_model(self):
        from toolkit.output.charts import heatmap_strategy_model
        data = [[0.05, 0.03], [0.08, 0.06]]
        fig = heatmap_strategy_model(data)
        assert fig is not None

    def test_phase_timeline_dataclass(self):
        from toolkit.output.charts import phase_timeline_chart
        fig = phase_timeline_chart([_FakeResult()])
        assert fig is not None
