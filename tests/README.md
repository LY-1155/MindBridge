# 自动化测试说明

类比 Java 的 `mvn test`，本项目使用 **pytest** 在**不启动独立 uvicorn 进程**的情况下，通过 **FastAPI TestClient** 在内存里调用 API。

## 安装（首次）

**先单独装 pytest**（体积小，避免拉全项目依赖时卡在第三方编译）：

```bash
python -m pip install -r requirements-dev.txt
```

或直接：

```bash
python -m pip install pytest
```

运行 `tests/` 里的用例会 **import `api.main`**，一般需要已安装业务依赖：

```bash
python -m pip install -r requirements.txt
```

若安装 `requirements.txt` 时在 **`editdistance`** 等处编译失败：常见于 **Python 3.14** 过新、依赖尚无预编译 wheel。可改用 **Python 3.11 / 3.12** 新建虚拟环境后再装依赖；或暂时注释掉本项目未用到的重型包（如仅跑契约测试时可先不装 `funasr` 等），待团队统一环境。

（Windows 下若直接输入 `pytest` 提示找不到命令，请始终用 **`python -m pytest`**。）

## 运行全部测试

```bash
python -m pytest
```

不要使用单独的 `pytest` 命令，除非你已把 Python 的 `Scripts` 目录加入系统 PATH。

## 只跑部分用例

```bash
python -m pytest tests/test_api_intervention.py -v
python -m pytest tests/test_api_pipeline_parallel.py -v
python -m pytest tests/test_pipeline_orchestrator.py -v
```

## 配置

- `pytest.ini`：测试发现路径、安静模式等
- `tests/conftest.py`：项目根路径、`api_client` 夹具

## 与 Postman 的区别

这些用例用代码固定请求体（`schemas/contracts/samples/` 下的 JSON），断言状态码与 JSON 结构，适合 CI 与回归，不必手工点 Postman。

## 为什么业务「还没做完」测试却全绿？

`tests/test_api_intervention.py` 等用例在 **Mock/Stub 或契约层** 上断言：例如默认 `MOCK_INTERVENTION=true` 时，只检查返回里是否有 `[mock-comfort]` 等**约定好的占位内容**，**不**检查真实大模型、RAG 或危机流程是否已接好。因此 **通过 = 接口与契约 + 当前假实现行为正常**，**不等于** 产品功能已完整实现。实网/实模型能力需另做集成测试或人工验收。

## 打印单测的输入与输出

对 `test_api_intervention` 等，**推荐**使用自定义参数（无需在 PowerShell/CMD 里纠结 `set`）：

```powershell
python -m pytest tests/test_api_intervention.py -v -s --print-io
```

仍可用环境变量（必须加 **`-s`**，否则 print 会被 pytest 吞掉）：

```powershell
$env:PYTEST_PRINT_IO="1"; python -m pytest tests/test_api_intervention.py -v -s
```

请勿在 **PowerShell** 里使用 CMD 写法 `set PYTEST_PRINT_IO=1 && ...`，否则会报错；若要用 `set`，请在 **cmd.exe** 中执行。

## 根目录旧脚本去哪了？

原先散落在仓库根目录的 `test_api.py`、`test_multimodal.py`、`verify_test.py` 等已迁至 **`scripts/integration/`**，见该目录下的 [`README.md`](../scripts/integration/README.md)。它们多为**手动冒烟 / 依赖服务**，与 `pytest` 分离，避免混淆。
