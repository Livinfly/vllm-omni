# vLLM-Omni FastH3 V2 性能测试

沿用 workspace 其他框架的命名配置、GPU 等待/锁、后台启动、状态、监控和报告协议。
脚本独立于 `exp/`、FastVideo 和 SGLang；只支持原生 FastH3 V2 完整 checkpoint。
H3 attention 的 topk 已修正为按 checkpoint sparsity 动态计算，与 FastVideo/SGLang 一致。
其余沿用原生 pipeline、attention kernel、scheduler、TE/VAE 和 offload 实现。
通过 Omni 支持的 `worker_extension_cls` 安装计时 hooks，并在请求外用控制 RPC 准备/收集指标。
原生推理方法仅加异常记录包装，参数、返回值和异常原样传递，不增加同步。

## 环境与启动

在 vllm-omni 仓库根目录、`exp-fixed-version` 分支执行：

```bash
bash benchmarks/video_generation/setup.sh
# 发布目录缺少原生 VAE provenance 时，先校验并准备独立 checkpoint；不修改源文件。
.venv/bin/python benchmarks/video_generation/prepare_models.py \
  --source /models/FastVideo-FastH3-8-Step-V2 \
  --base-model /models/MiniMax-H3 \
  --output .cache/benchmark/models/fasth3_v2_verified
source .cache/benchmark/models/fasth3_v2_verified/environment.sh
# 先确认原生实际配置、解析后的形状和离线 VAE 资产，不加载权重。
.venv/bin/python benchmarks/video_generation/preview.py \
  --model-path .cache/benchmark/models/fasth3_v2_verified --gpus 8 4 2 1 --output preview-v2.json
.venv/bin/python benchmarks/video_generation/run.py \
  --model-path fasth3_v2=.cache/benchmark/models/fasth3_v2_verified \
  --gpus 8 4 2 1 --output outputs/benchmarks/omni-v2 --background
.venv/bin/python benchmarks/video_generation/monitor.py outputs/benchmarks/omni-v2 --watch
.venv/bin/python benchmarks/video_generation/report.py outputs/benchmarks/omni-v2
```

需要 Python 3.12、uv、兼容 torch 2.13 / CUDA 13 的驱动与 GPU。依赖完整锁定在
`requirements.lock`；setup 不安装历史实验插件。`UV`、`BENCH_VENV`、`PYTHON_VERSION` 可覆盖；
`setup.sh --check` 检查已有环境并记录 freeze。上游 optional CUDA 扩展 wheel 还需兼容宿主 glibc；
必要时用 `FASTVIDEO_KERNEL_PACKAGE=/compatible/fastvideo_kernel-0.3.4.whl` 指定相同版本的兼容构建。
`BENCH_CUDA_COMPAT` 可显式指向已安装的 CUDA compat 目录，脚本不修改系统驱动。
媒体封装使用原生 `mux_video_audio_bytes`（PyAV）。

## 模型与推理配置

`--model-path NAME=PATH` 指定完整本地原生可加载 snapshot。须含 transformer / text_encoder
权重与索引、tokenizer、processor、`model_index.json`、`fastvideo_inference.json` 和
`provenance.json`。权重来源为 `FastVideo/FastVideo-FastH3-8-Step-V2`，metadata 必须声明 V2。

原生 `FastH3CheckpointSpec.resolve_native_vaes` 从 provenance 的
`hf://MiniMaxAI/MiniMax-H3@<40位commit>` 定位基础模型的 `FL2VA/video_vae`、`FL2VA/audio_vae`。
这些资产必须事先缓存于 Hugging Face 默认 cache（可用标准 `HF_HOME` / `HF_HUB_CACHE` 指定）。
原生 H3 VAE 通过 checkpoint 的 modeling code 加载，启动配置显式设置 `trust_remote_code=True`。
启动和 preview 均固定离线，不下载或把 FastVideo VAE 冒充原生 VAE。
即使 snapshot 含 `vae/`、`audio_vae/`，原生路径仍会解析 pinned base VAE。
本次检查发布者 `main/provenance.json` 返回 HTTP 404。`prepare_models.py` 提供本地校验方案：
从基础模型自身的 `.cache/huggingface/download/*.metadata` 读取确切 commit，校验 VAE 配置、代码和权重的
ETag，再逐张量比较它们与 V2 的视频/音频 VAE 及归一化参数。视频导出的 QKV/FF 布局和名称会先还原，
仅省略导出未保留、已验证全零的 decoder mask token。缺文件、来源记录或任何差异均拒绝准备。
脚本只用 CPU 流式读文件，不执行 checkpoint 代码。

输出通过符号链接复用源 checkpoint，补齐原生 `model_index.json`，生成标明
`publisher_provided=false` 的本地 VAE binding provenance、`vae_verification.json` 和离线 HF snapshot。
这份记录证明本次推理的 VAE 绑定，不声称是发布者或训练来源。源目录保持原样，输出目录拒绝覆盖。
当前本地验证的基础 commit 为 `9bfb6693f2cf6de171db46d1aa586f67d773a1da`；
视频 560 个原生张量对应 703 个导出张量，音频 1087 个张量，全量内容比较通过。
准备后须加载输出的 `environment.sh`；preview/worker 继续用原生解析器检查离线 VAE。

默认配置在 `configs/fasth3_v2.json`：同 FastVideo café prompt，seed1000、1344×768、124 帧、
24 FPS、CFG1、8 次完整 DiT forward（9 个 sigma 节点），DMD 阶梯
999/874/749/624/500/375/250/125，video/audio shift=10/3；FastVideo provider / BF16 / H3 tile64。
TP1、Ulysses=卡数、ring1、allgather1、CFG parallel1。未显式覆盖的 TE/VAE 并行、HSDP、offload、
pinning、compile、cache 等全部采用当前 Omni 原生默认，不强行对齐其他框架。
原生阶段 profiler 关闭以避免额外同步；profile 不允许覆盖计时扩展/拓扑/采样 contract。
可复制配置修改命名、工作量及 `engine` 支持的显式策略，实际配置逐 rank 保存。
`--prompt` / `--prompt-file` 可覆盖 prompt；`--config` 可传多个命名配置。

**topk 对齐**：`MiniMaxH3VSAImpl` 的 H3 路由读取模型传入的 `vsa_h3_sparsity`，
计算 `max(1, min(ceil((1-sparsity) * video_blocks), video_blocks))`。
默认画布对应 660 个 video blocks，V2 sparsity0.8 的实际 topk 为132，prefix 始终 dense。
缺少 sparsity metadata 的旧调用保留 backend 默认 topk；不改变 backend 配置本身。
harness 逐 rank 分别记录实际 topk、backend 默认值、sparsity、provider/precision 和 fallback 策略。
配置检查不是每次 kernel 执行的证明，需同时查看原生路由/fallback 日志。

默认每组 `dit` 模式，1 次完整预热 + 1 次正式测量，`--gpus` 默认 1。
`--modes dit e2e --repeats 3` 可做双模式对照；每模式单独预热，正式请求轮流 AB/BA。
`--dry-run` 只显示配置和命令，不访问权重/GPU或创建 suite。

## 计时边界

- `response_e2e_s`：客户端进入 `Omni.generate` 前至原生音视频封装和 MP4 写盘返回。
  含 IPC、TE、噪声、DiT/scheduler/通信、VAE、返回转换、编码封装、保存。
  不在返回后增加 CUDA 同步；`api_only_s` 另存，但不替代含保存 E2E。
- `dit`：根 transformer 完整 forward 的 CUDA events；开始 hook 在已有 root pre-hooks 前，
  结束 hook 在已有 root post-hooks 后。包含 hook 内的 SP/FSDP/offload，不含 scheduler。
  事件在请求外预分配/初始化，不逐步同步。主指标各 rank 先累加再取最大 `max(sum)`，
  另存 `sum(max)` 和每 rank/step 原始值。要求实际恰好 8 次完整 forward。
- `e2e`：没有 DiT hooks；纯 e2e suite 也不分配 CUDA events。两模式控制 RPC 相同。
- `distributed_e2e_s`：同一客户端起点至保存后发起的收集 RPC 到达各 worker、完成 Gloo barrier。
  包含额外 RPC 调度开销，可能显著高于 response E2E；不能直接与其他框架的 barrier 指标相比。
  仅支持本机多卡，使用同机 monotonic clock；所有逐 rank 数据保留。
- 模型加载、设备校验、等待、准备、预热、事件查询、指标聚合、媒体解码及 hash 排除。
  每次请求前重置 Python/NumPy/torch/CUDA seed，采样显式带 seed；不宣称跨框架噪声或视频相同。
  每次请求计时后完整解码验证尺寸/帧数/FPS/音轨和音频有限值，`--validate-video` 再记录文件 SHA256。

## 状态、等待和报告

`CUDA_VISIBLE_DEVICES` 接受 nvidia-smi 数字索引或唯一 UUID 前缀，worker 加载前核对真实 CUDA UUID。
目标卡有计算进程、显存 >256 MiB 或利用率非零时等待，间隔30秒，连续两次空闲后启动。
与其他框架共用 `/tmp/video-benchmark-gpus-UID/` UUID 文件锁，持有到组进程结束；推理期间不轮询 GPU。
外部程序仍可占卡；不支持 MIG。`--wait-for /other/status.json` 等待前置 suite。

每次保存 plan、scripts/配置快照、源码 commit/diff、小配置 hash、权重大小、逐 rank 有效配置、
metrics、MP4、日志、状态和报告；已有输出目录拒绝覆盖。运行中不要修改推理源码、模型或快照。
原生 VAE 的配置/代码 hash、权重大小也保存在有效配置中；离线 cache 缺配置、权重或 shard 时提前报错。
原生权重/编译失败或 OOM 不自动重试或降配。加载/预热阶段明确的 rank CUDA OOM 记录后继续下一组；
正式 OOM 和普通错误停止矩阵。原始错误与阶段保存在 `error_rankN.json` / `error_client.json`，
`status.json` 展示根因；后台 launcher 日志位于 suite 同级。编译缓存隔离到每组每 rank。

`report.md` / `summary.json` 按组更新，包含正式/预热耗时、同配置同模式单卡加速比、双模式对照、
波动提示；单次 stdev=null。历史比较默认关闭；`--baseline /old/summary.json` 显式开启，schema、
源码、模型配置或有效策略不一致时不计算百分比。schema5 表示 Omni 独有的 E2E/RPC 边界。

CPU harness 回归：`python3 -m unittest discover -s benchmarks/video_generation -p 'test_*.py' -v`（Python3.10+）。
原生 attention 回归：`python -m pytest tests/diffusion/models/minimax_h3/test_vsa_layout.py tests/diffusion/attention/test_fastvideo_vsa.py -q`。
测试包含小张量 CPU attention/block-map 校验，CUDA kernel 使用替身；不加载模型。
完整 GPU 推理通过前不能声称本配方已实测。

## 后续支持 V1 需要什么

1. 扩展完整 checkpoint 准入：当前 `FastH3CheckpointSpec` 只接纳 V2。需要校验 V1 model_id、
   metadata、训练 gate、权重映射和来源；不能仅改模型名或换成另一个 LoRA。
2. 参数化原生 checkpoint sampling contract：4 次 forward、5 个 sigma 点、阶梯999/749/500/250、
   video/audio shift12/3、sparsity0.9（默认画布 topk66）。当前 V2 的 8 步、shift10/3 检查需按版本区分。
3. 解析 V1 provenance 对应的原生 VAE，核实完整权重、TE/tokenizer/processor 兼容性。
4. 实际核查 gate、SP/VSA 路由、topk、scheduler sigma 和输出媒体，至少完成 G1/G8 的完整预检，
   再跑矩阵及双模式计时开销检查。

动态 topk 已由原生 H3 路由共享支持，V1 仍需完成以上准入和采样适配。
历史插件还调整过 TE 每卡复制/FSDP CPU offload、VAE 驻留及部分融合；这些策略对齐不是支持 V1 的必选项。
本次不引入该实验插件，也不依赖历史实验目录运行。

## Sparse attention runtime 计时（scope v2）

添加 `--sparse-attention`（需 `dit` 模式）。起点是 attention 输入通信阶段开始前；终点是最后结果回传/聚合及必要依赖等待完成、输出可用时。区间内包含路由/选块、调度、稀疏计算、KV/元数据通信、结果合并与等待。单卡采用对应 runtime 的入口/出口。

通常不含输入 Norm、通信前的 QKV/QK Norm/RoPE、输出投影和 residual；按源码实际位置计入，不能笼统宣称所有框架这些操作都在区间外。SGLang H3 包含反向 All-to-All 后的 coarse All-Gather 和 gate fold；SSTA 包含文本头 All-Gather；NABLA 多卡通信与投影交错，连续区间会包含 K/V 投影和 QK Norm/RoPE，manifest 明确标注。准确起止点见 [sparse_attention.py](sparse_attention.py) 的运行时 `boundaries`。

事件在请求外预分配并初始化，不新增逐层同步或 collective；异步通信利用原有执行流依赖保证结果可用。保留原生 hooks/编译 helper；NABLA 多卡在 DTensor hooks 内的事件记录显式保持 eager，可能产生 helper 图断点，需单独核实插桩扰动。whole-block compile/graph 不支持，不自动改变原配置。

逐 forward/layer/rank 数据、源码锚点和 `scope_version=2` 保存于 `metrics.json` 的 `per_rank[].sparse_attention`，计划文件也记录版本和范围。主拆分在完整 DiT 最慢的同一 rank 上计算：attention + 其余 DiT = 完整 DiT；另存最大 attention rank 的总和，不相加各层独立最大 rank。

旧 `scope_version=1` 是 Norm → residual，不能重标或混入新结果。`dit` 模式在同一请求记录 E2E、完整 DiT 和 attention 三项；使用 `--modes dit --warmup 1 --repeats 1 --sparse-attention` 可做单次测量。`e2e` 模式移除 attention 插桩，可按需作对照；模式间差异包含整请求波动，不等同于纯插桩成本。各模型的 GPU 验证状态以具体实验结果为准。
