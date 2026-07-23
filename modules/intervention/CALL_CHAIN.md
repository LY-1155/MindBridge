# 干预闭环 · 调用链（只读）

```
用户 / 客户端
    │
    ├─► POST /api/v1/modules/intervention/run    … api/routes/parallel_modules.module_intervention_run
    │       └── get_pipeline_services(settings).intervention.intervene(body)
    │
    └─► POST /api/v1/pipeline/run               … api/routes/pipeline.pipeline_run
            └── pipeline.orchestrator.run_pipeline(inp)
                  └── （先 safety → emotion → route）
                  └── 组装 InterventionRequest → svc.intervention.intervene(intervention_req)

装配与实例来源：

    modules.runtime.get_pipeline_services
        └── modules.factory.build_pipeline_services
                  └── get_intervention_service(settings)   # MOCK_INTERVENTION → Mock / Stub

契约：

    schemas/contracts/v1.py  … InterventionRequest / InterventionResult

实现入口（当前）：

    modules/intervention/mock.py   … MockInterventionService.intervene
    modules/intervention/stub.py   … StubInterventionService.intervene

协议抽象：

    modules/ports.py  … InterventionPort
```
