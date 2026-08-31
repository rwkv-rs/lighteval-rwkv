## 核心目标
LightEval 是 LLM 社区的主流评估库, 本仓库需要通过 LightEval 原生 pipeline、model adapter 和 task 接口完成 RWKV 模型的接入.
代码原则: 每一个文件/类型/函数/变量都需要找到相似实现作为原型, 若该原型带有模型名则将其替换为 `RWKV` 或其它大小写变种, 否则保持同名.

## 目录规范

```text
configs/eval/                    RWKV 一键评估配置与外部端点池 manifest schema 示例
src/lighteval/main_rwkv.py       RWKV 一键评估 CLI 和配置校验
src/lighteval/models/rwkv/       RWKV HTTP model adapter 与端点池客户端
src/lighteval/tasks/             LightEval 原生 benchmark 定义; 不因目标清单缺项而补 task
tests/unit/rwkv/                 RWKV CLI、配置、HTTP pool 和 model 的 hermetic 测试
temp/                            外部部署 manifest、启动脚本与四模型进程编排入口
```

`src/lighteval/`、LightEval task 和输入契约内的改动必须应用对照原则；`temp/` 不属于 LightEval 管理范围，只存放外部系统提供的部署快照和操作入口，不以 LightEval 内部实现作为代码原型.

model_name 需要写清楚 Qwen(如 Qwen3.5-2B ) / RWKV7 (详情见 `RWKV7 权重` 一章节) 权重具体版本号.
新增任何文件, 都需要得到用户确认.

## 权威 RWKV7 实现
(1) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
(2) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py
(3) https://github.com/BlinkDL/Albatross -- 权威底层推理引擎实现仓库 (cuda, for pro6000, 无调度, 无varlen)
(4) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp -- 权威预训练实现仓库 (cuda, for h100)
(5) https://zhiyuan1i.github.io/posts/dplr-mathematics -- Diagonal Plus Low Rank(DPLR）的数学原理：显式转移矩阵的并行计算
(6) https://github.com/rwkv-rs/transformers-rwkv/tree/rwkv -- 权威 RWKV Huggingface Transformers 适配仓库 (with rust tokenizer, x10 faster than python implementation)

## RWKV7 权重
权重一般命名规范: {arch_version}-{data_version}-{param_size}-{release_date}-{ctx_len}.pth
如: rwkv7-g1h-7.2b-20260710-ctx10240.pth
arch_version: 架构版本, 如 rwkv7(default), rwkv7a(experimental, rwkv7 with DeepEmbed), rwkv7b(experimental, rwkv7 with DeepEmbedAttn)
data_version: 数据版本, 如 g1a, g1b... (The further back in the alphabet, the better)
param_size: 参数规模, 仅有 0.1b, 0.4b, 1.5b(often used in RL), 2.9b, 7.2b(often used in the infer test), 13.3b
(1) https://huggingface.co/BlinkDL/rwkv7-g1/tree/main -- 权威权重 Release 源 (update every month)
(2) https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main -- 权威权重 Test 源 (不定期update)
(3) https://huggingface.co/rwkv-rs/rwkv7-g1-st -- 权威权重 Release 源 (for transformers)

## 正确性检查
1. 是否能够正确应用 transformers-rwkv 以及对应 rwkv7-g1-st 权重仓库中提供的三组 Prompt Template
2. 默认使用 wkv_mode=fp32io16
3. 当使用 Open Think 模式时, 使用解码参数 temp 0.96, top_p 0.76, top_k 32, presence_penalty 1.0, frequency_penalty 0.1, penalty_decay 0.988; 使用 Fake Think 模式时, 使用解码参数 temperature 1.0, top_p 0.28, top_k 32, 使用 Open Think + Function Call 模式时, 使用解码参数 temp 0.96, top_p 0.76, top_k 32, 关闭 penalty.
4. 参考 https://github.com/BlinkDL/Albatross/blob/main/faster3a_2605/eval_gpqa_diamond.py 完成选择题的通用判分器实现
5. 参考 https://github.com/BlinkDL/Albatross/blob/main/faster3a_2605/eval_math500.py 完成简答题的通用判分器实现
6. 模型分数应当于 Qwen3.5 相似参数量模型有相似的得分

## 吞吐量检查
1. 显存余量应当小于总量的 10%, GPU 利用率应达到 97%, 如 transformers-rwkv 或 FlashRWKV2 存在性能问题, 请及时反馈给用户.
2. 使用 Http 协议支持评估端与推理端的通信, 请求并发量应当略大于推理并发量(允许少量排队, 但禁止空载)

## 结果保存
记录详细 (benchmark_name, model_name, n_samples, k_metrics, cot_mode, prompt_template), [_可选完成 wkv_mode fp32io16 vs fp16 对比] 对应的 (正确率, 截断率) , 其中截断率定义为达到输出上限未能完成作答的样本数 / 总样本数

## 职责边界
本仓库独立负责当前已注册 benchmark 的 LightEval 原生评估流程、RWKV HTTP model adapter 和标准 results/details 输出. 外部系统负责推理服务生命周期、权重与 wkv_mode 切换以及端点池 manifest; 其它框架负责 LightEval 未注册的 benchmark. 本仓库不承载跨框架调度、外部评估生命周期或分数发布逻辑.

## Env
使用 uv 管理本机和远端专属环境 ./.venv, 严禁本项目使用其它环境, 严禁其它项目使用本项目环境, 避免环境污染问题。

## Inference API
url: api.rwkv.rs
1.5B: bsz1024
2.9B: bsz1024
7.2B: bsz960
13.3B: bsz320
