# RSI 配置入口

配置按用途归类，文件存在不代表当前可以执行。每份用途及状态见 catalog.json，统一可视化的“配置目录”也读取同一目录表。

- tasks/：显式执行模板，分 preflight、training、lifecycle、synthesis。须先完成输入、环境及预算预检。
- skills/：教师与学生等服务绑定。
- studies/：配对实验规则。
- policies/：来源隔离约束；测试日志清单内容不进入可视化。
- registry/：服务目录，描述已经注册的实现。
- examples/：历史场景、修复、恢复与实验示例。路径或证书可能已过期，不在默认推荐入口中。
- catalog.json：配置的用途、分类和状态索引。

已删除无人引用的 chainflow_navtrain_mining.json。此前用户停止 navtrain 学生全量评分的要求继续生效，界面不提供重新启动入口。
FPD 的训练模板仅作为不具备原生训练器的历史示例保留，不能标记训练已接通。
旧路径、旧任务回执不改写；重新执行须按新配置准备，任务ID、原生模型、loss与准入门槛保持原样。
