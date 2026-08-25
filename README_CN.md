# Causal Dialog Route Reranker 替代提交

这是 SE team 针对主办方反馈整理的严格合规优先版本。它只运行一条在线轨迹：
Navigator 独占并调用 QG，Guide 只做问题定位和答案生成，两个代理之间只传递
自然语言问题与回答。Guide 内部保存上一轮答案对应的路线状态，用它对下一轮
GTL top-5 定位候选做因果重排；该结构化状态不会传给 Navigator。

本版本不使用 QFP、目标描述共享、语言 DFS、视觉签名、双轨迹选择、外部 API
或评测标签。真实 released-Test SR 为 27.3684%（78/285），隐藏 Test Score
未知。提交文件 SHA256 为
`52c191922ff14423efe5b290c0a9e1aa0fa0e094c06227145db3a421fbbf5efc`。
