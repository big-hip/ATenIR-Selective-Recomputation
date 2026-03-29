# legacy diagnostics

该目录用于存放不属于当前主链（meta-first 捕获/仿真）的历史诊断脚本。

当前迁入：

- `quick_diag.py`
- `diag_compile.py`

保留目的：

- 避免删除历史排障代码。
- 保持 `examples/transformer` 根目录聚焦主流程。

若后续需要重新纳入主链，请先完成：

1. 明确功能边界（capture / benchmark / train 中哪一条）。
2. 合并重复逻辑，避免再次散落脚本。
3. 在 `WORKFLOW_META_FIRST.md` 中补充使用说明。
