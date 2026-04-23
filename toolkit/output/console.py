from toolkit.utils import format_bytes, normalize_row

try:
    from tabulate import tabulate
except Exception:
    tabulate = None


_BYTE_KEYS = frozenset({
    "param_bytes", "grad_bytes", "bw_fw_delta", "optimizer_bytes", "activation_bytes",
    "estimated_peak", "peak_allocated", "peak_reserved", "base_allocated",
    "activation_delta", "fw_peak", "bw_peak", "saved_act_bytes",
    "fw_output_bytes", "total_alloc_bytes", "total_view_bytes",
    "runtime_peak", "runtime_reserved", "static_peak",
    "fw_peak_bytes", "bw_peak_bytes", "act_peak",
    "opt_peak", "fwbw_peak", "true_peak", "opt_temp",
    "fw_graph_peak", "bw_graph_peak",
    "base", "after_fw", "after_bw", "after_opt", "overall_peak",
    "l1_true_peak",
    # L2 explicit keys
    "l2_fw_peak", "l2_bw_peak", "l2_opt_peak", "l2_fwbw_peak", "l2_true_peak",
    # L2.5 fusion-aware keys
    "l25_fw_peak", "l25_bw_peak", "l25_opt_peak", "l25_fwbw_peak", "l25_true_peak",
    "l25_fusion_fw_peak", "l25_fusion_bw_peak", "l25_fusion_opt_peak",
    "l25_fusion_true_peak",
    "l25_safe_fw_peak", "l25_safe_bw_peak", "l25_safe_opt_peak",
    "l25_safe_true_peak",
    # L3 scheduler keys
    "l3_fw_peak", "l3_bw_peak", "l3_opt_peak", "l3_fwbw_peak", "l3_true_peak",
    "sched_fw_peak", "sched_bw_peak",
    # Runtime explicit keys
    "rt_fw_peak", "rt_bw_peak", "rt_opt_peak", "rt_true_peak",
    "shape_sum_fw_bytes", "shape_sum_bw_bytes", "shape_sum_true_peak",
})


def _stringify(value, key=None):
    if isinstance(value, int) and key in _BYTE_KEYS:
        return format_bytes(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def print_comparison_table(results: list[dict], title: str = "Comparison"):
    rows = [normalize_row(result) for result in results]
    if not rows:
        print(title)
        print("(no results)")
        return

    headers = list(rows[0].keys())
    table_rows = [[_stringify(row.get(header, ""), key=header) for header in headers] for row in rows]

    print(title)
    if tabulate is not None:
        print(tabulate(table_rows, headers=headers, tablefmt="github"))
        return

    widths = [max(len(str(header)), *(len(str(row[idx])) for row in table_rows)) for idx, header in enumerate(headers)]
    header_line = " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers))
    sep_line = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(sep_line)
    for row in table_rows:
        print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))


def print_step_result(result):
    row = normalize_row(result)
    print_comparison_table([row], title=row.get("name", "StepResult"))
