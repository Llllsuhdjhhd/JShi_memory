# 开发与验证

## 配置

配置模型位于 `src/rems/config.py`。环境变量前缀为 `REMS_`，嵌套配置使用双下划线，例如 `REMS_LLM__API_KEY`。项目提供 `.env.example` 作为配置示例；复制为本地 `.env` 后填写实际值。不要把密钥提交到仓库。

存储配置包括关系数据库 URL 和 Qdrant URL/本地路径。实际部署前，确认持久化路径、向量集合名和嵌入模型版本与当前环境相匹配。

## Miniconda 离线测试环境

仓库根目录的 `environment-test.yml` 定义名为 `jshi-memory-test` 的 Python 3.12 测试环境。请从已初始化 Conda 的终端（例如 Miniconda Prompt，或已运行过 `conda init powershell` 的 PowerShell）在仓库根目录执行：

```powershell
conda env create -f environment-test.yml
conda activate jshi-memory-test
pytest tests/unit -q
pytest tests/integration/test_jshi_recall_contract.py -q
```

第一条创建环境，第二条激活环境，后两条分别运行离线单元测试和与相邻 JShi 仓库的契约测试。契约测试在找不到 `../JShi/src` 时会跳过。单元测试可以使用假 LLM、哈希嵌入和临时数据库，不需要真实模型 API 或下载语义嵌入权重。

如果终端提示 `conda` 无法识别，通常是当前 PowerShell 尚未加载 Conda 初始化脚本；先使用 Miniconda Prompt，或在 Miniconda 安装终端运行一次 `conda init powershell` 后重新打开 PowerShell。若 `pytest` 无法识别，先确认 `conda activate jshi-memory-test` 成功。

## 测试边界

- `tests/unit/`：离线验证模块行为。
- `tests/integration/test_jshi_recall_contract.py`：有相邻 JShi 仓库时验证端口兼容，不请求在线模型。
- `tests/integration/` 与 `tests/scenarios/`：按具体用例可能需要更完整服务或模型配置；先看用例本身的前置条件。
- `tests/_archive/` 的旧材料已移入 `docs/history/`，不作为当前 pytest 测试入口。

当前文档重整没有执行测试。后续改动如需验收，应在对应 Conda 环境运行相关测试并把结果记在草稿的验证区，完成后同步到模块文档。
