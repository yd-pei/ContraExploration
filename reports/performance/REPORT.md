# Contra Rainbow 训练性能实测与优化

2026-10-20，Apple M4 MacBook Air（4P+6E CPU、10 核 GPU、24 GiB 内存），uv / Python 3.12 / PyTorch 2.14.1。

CPU 从 **70.31 → 77.96 决策步/秒（+10.9%）**；MPS 从 **64.43 → 两次平均 79.66（+23.6%）**。优化后的 CPU 和 MPS 吞吐接近；MPS 主要还能释放 CPU 资源。所有主要对照均为 2,048 个决策步、449 次参数更新，没有启动长期训练。

## 同预算端到端结果

| 配置 | 决策步/s | 实际模拟器帧/s | 更新/s | 进程峰值 RSS MiB | 进程平均 CPU |
|---|---:|---:|---:|---:|---:|
| CPU 原始，4线程 | 70.31 | 281.13 | 15.41 | 542.5 | 275.3% |
| CPU 优化，4线程 | 77.96 | 311.74 | 17.09 | 626.0 | 271.0% |
| MPS 原始 | 64.43 | 257.55 | 14.13 | 539.9 | 30.8% |
| MPS 优化，第1次 | 79.58 | 318.19 | 17.45 | 603.4 | 35.9% |
| MPS 优化，第2次 | 79.74 | 318.96 | 17.48 | 602.8 | 36.6% |

CPU 的 100% 表示一个核，275% 约 2.75 个核。峰值 RSS 是当前进程整个生命周期的 `getrusage` 高水位，包含构建模型的内存。MPS 分配信息另见原始 JSON，不能与 RSS 简单相加。

固定设置：seed 4、warmup 256、batch 32、replay capacity 20,000、每 4 步更新一次、3-step、gamma .99、51 原子、支持 [−10,10000]、同动作/预处理/Level1、首命结束、episode 上限 1,000 决策步、相同目标同步频率、同 reward mode progress。原始 CPU/MPS 保留压缩回放、物化噪声矩阵、循环 Adam；CPU 优化用 raw 回放和因子化 actor，Adam 仍用 loop；MPS 优化额外显式启用 fused Adam。

计时覆盖训练循环内的模拟器、预处理、推理、回放、更新、日志和最终检查点，不包含 Python 导入、环境/模型构建；MPS 在计时起止同步。计时和资源采样的开销在两边均开启。各决策实际帧数因提前死亡可能略有不同，所以单独报告真实帧/s。

主要结果使用互不重叠的正常进程。早期 `base-cpu1` 与 `opt-cpu4` 探索曾短暂重叠，未用于最终速度结论；后者已由 `opt-cpu4-clean` 独立复测替代。MPS 两次优化复测仅相差约 0.2%。单机背景负载未完全隔离，也未做数小时的散热/热降频测试。

## 实测瓶颈与改动

原 CPU 的学习更新占约 78%：449 次更新合计 22.76 秒，其中前向与分布投影 14.82 秒、反向/梯度裁剪 4.30 秒、优化器 2.88 秒。模拟器加预处理 1.85 秒，日志约 .013 秒，最终检查点 .16 秒。日志和模拟器不是主要瓶颈。

1. **单帧 NoisyNet 推理的等价因子化实现**。对 actor 使用 `linear(x, μ, bμ) + linear(x * εin, σ, bσ) * εout`，避免每个动作构造庞大的 `μ + σ * outer(εout, εin)` 矩阵。每步仍重新采同样分布的噪声；学习过程仍用原物化实现。CPU 推理总耗时从 3.95 降至 1.76 秒。可用 `--standard-actor` 还原原实现；新向量缓冲不进入 state_dict，原模型键保持兼容。
2. **可选 raw uint8 环形回放**。去掉 zlib 压缩/解压，以连续数组批量 gather；保持相同 PER、随机采样、n-step、重要性权重和观测内容。CPU 回放写入 .376 → .041 秒，采样 .611 → .076 秒。`--replay-storage raw` 为速度选择，默认仍是省内存的 zlib。raw 容量满 20,000 时，仅双份观测约 1.13 GB（1.05 GiB）；此处短测只填约 2,046 条，表中 RSS 不代表满回放内存。
3. **MPS 显式选择 PyTorch fused Adam**。保持学习率、epsilon、betas 和梯度裁剪。同步阶段测量中优化器从 9.30 降至 4.72 ms/更新。CPU 的 foreach/fused 筛选未见收益，所以 CPU 推荐仍为 loop。这个性能选择不会跟 reward mode 自动切换。
4. **新增可重复监测入口**。`--profile path.json` 采集阶段耗时、实际步/帧/更新吞吐、进程 CPU、RSS 高水位和系统 GPU 计数器；`--profile-sync` 用于短程诊断，不能把其额外屏障开销当成常态吞吐。

1/2/4/8 线程及 contiguous/channels-last 小型筛选未发现值得采用的布局收益；4 线程仍为推荐。没有扩大 batch 或增加多环境来人为提高占用率，这些会改变研究比较的训练参数/数据交互。主要瓶颈仍是学习计算，当前改动保持顺序采样与更新比例。

## MPS 分阶段同步诊断

以下是另外的 256 步、warmup 32、56 次更新短测，阶段边界调用 [torch.mps.synchronize](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html)。它用于分解耗时，不与上表未加内部屏障的吞吐混用。单位毫秒；父级/子级计时存在嵌套，不能把所有计时求和。

| 阶段 | 计时单位 | 原始 ms | 优化 ms |
|---|---|---:|---:|
| inference | 每决策步（含随机 warmup） | 3.935 | 2.291 |
| emulator_step | 每模拟器帧 | 0.206 | 0.210 |
| preprocessing | 每决策步 | 0.163 | 0.173 |
| replay_add | 每决策步 | 0.226 | 0.043 |
| replay_sample | 每次更新 | 1.763 | 0.197 |
| tensor_transfer | 每次更新 | 2.016 | 1.805 |
| learn_forward_and_projection | 每次更新 | 11.018 | 12.071 |
| backward_and_clip | 每次更新 | 19.639 | 19.522 |
| optimizer | 每次更新 | 9.297 | 4.719 |
| priority_update | 每次更新 | 0.643 | 0.621 |
| logging | 每条日志 | 0.068 | 0.053 |
| checkpoint | 每次检查点 | 191.326 | 199.830 |

未同步的 MPS 计时中，`priority_update` 看似很慢，实际包含等待之前排队 GPU 工作的时间；同步诊断显示真正 PER 更新约 .6 ms，不能据此误判 PER 是 GPU 学习的主要瓶颈。

## CPU/GPU 利用率与权限

- 沙箱内 PyTorch：MPS built=True、available=False；正常进程：available=True，Metal 矩阵运算验证通过。见 [正常进程探测](accelerator-normal.json) 和 [受限上下文探测](accelerator-sandbox.json)。这是执行上下文差异，不能据此说 M4 不支持 MPS。
- GPU 指标来自免 sudo 的 IORegistry `Device Utilization %`，约每 .5 秒采样，是**系统全 GPU**，不能当成训练专属进程利用率。原始 MPS 平均约 70.5%，优化后约 75.8%/76.0%，两次最高分别 100%/85%。CPU-only 基准期间系统 GPU 也有约 23%/37% 背景活动，进一步说明不能独占归因。
- CPU 指标由进程自身 user+system CPU 时间差计算；不会拿 CPU 数字冒充 GPU 利用率。MPS 优化进程平均仅用约 .36 个 CPU 核，CPU 学习约 2.71 个核。
- `powermetrics` 返回需超级用户；没有使用 sudo、安装系统服务或声称获得功耗/温度。GPU 分配器内存是结束时值，不是 GPU 峰值。

## 正确性

26 项测试在正常 Metal 上下文通过。覆盖因子化/物化噪声输出和梯度、GPU 推理数值、两种回放相同索引/数据/权重/覆盖/拷贝语义、Adam 更新及已有完整 Rainbow、三奖励与任务结束测试。

CPU 主对照中，449 次更新后的**全部模型 state_dict 张量逐位一致**，最大差为 0，见 [CPU 等价证据](cpu-equivalence.json)。MPS fused Adam 改变 FP32 运算融合/舍入次序：三步合成测试中两种后端参数最大差 1.49e−8，二阶矩相对差约 1.3e−5（FP32 中 `1−.999` 的舍入），两者参数与 float64 参考最大误差均约 2.16e−7。测试据参考结果对二阶矩使用 2e−5 相对容差；不宣称 GPU 两种后端逐位复现。详见 [数值核验](adam-numerics.json)。

三种奖励、动作、预处理、首命/全部生命语义、C51/PER/NoisyNet 等六组件保持不变。统一快速实验配置为三种奖励共用同一性能后端；没有新增此前仅讨论的定期评估、曲线或最佳模型功能。

## 推荐运行与复现

在有正常 Metal 权限的终端，进入项目目录：

```bash
uv run contra-rl train --device mps --threads 4 \
  --replay-storage raw --adam-backend fused \
  --steps 2048 --warmup 256 --batch 32 --capacity 20000 \
  --max-steps 1000 --seed 4 --output runs/my-performance-check \
  --profile runs/my-performance-check/profile.json
```

这复现短测预算，不启动数小时训练。仅需 CPU 时使用 `--device cpu --replay-storage raw --adam-backend loop --threads 4`；内存紧张时选 zlib。当前默认已启用等价快速 actor，其余性能选择显式记录在 config.json。

三奖励共用的推荐配置（默认只打印计划）：

```bash
uv run python -m contra_rl.experiments --config experiments/reward-comparison-fast-mps.json
```

用户决定正式执行时才添加 `--run`。基线代码冻结在 `reports/performance/baseline/contra_rl`，优化代码在 `optimized/contra_rl`，哈希在 `source-manifest.json`。可用同一 uv 解释器重新运行冻结基线：

```bash
uv run python scripts/benchmark_pipeline.py --source reports/performance/baseline train \
  --device cpu --threads 4 --steps 2048 --warmup 256 --batch 32 --capacity 20000 \
  --max-steps 1000 --seed 4 --output runs/my-frozen-baseline \
  --profile runs/my-frozen-baseline/profile.json
```

原始测量 JSON、采样序列、日志均在本目录；结构化汇总见 [summary.json](summary.json)。
