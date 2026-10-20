# 初代 NES《魂斗罗》Rainbow DQN：三种奖励对比

项目使用 **uv、Stable Retro 1.0.1、Gymnasium 1.4.0、PyTorch 2.14.1**。目标是初代美版 `Contra-Nes`，从 `Level1.state` 开始；ROM 已导入且哈希匹配。完整 Rainbow 六项组件、本地训练、评估和录像入口均已实现。当前仅完成执行验证，尚未训练出通关策略。

## 每种奖励的独立配置

可以直接编辑三份互不绑定的 JSON：

| 配置文件 | 奖励 | 默认输出目录 |
|---|---|---|
| [score_clipped.json](configs/score_clipped.json) | 原生游戏得分增量先累加，再取 sign | `runs/score_clipped-seed0` |
| [game_score.json](configs/game_score.json) | 原生真实游戏得分增量，不缩放/裁剪 | `runs/game_score-seed0` |
| [progress.json](configs/progress.json) | 现有进度塑形，效果待比较 | `runs/progress-seed0` |

在项目根目录运行相应的一条命令即可，**无需复制长参数串**：

```bash
uv run contra-rl train --config configs/score_clipped.json
uv run contra-rl train --config configs/game_score.json
uv run contra-rl train --config configs/progress.json
```

每条命令会按该文件开始训练；这里只提供入口，没有替你执行文件中的长期预算。三份默认均为 **1,000,000 次 agent 决策**（最多约 4,000,000 模拟器帧），使用已验证的 MPS / 4线程 / raw回放 / fused Adam / 快速 actor 设置。预算没有改成论文规模。请在正常 Metal 权限的终端运行；若用 CPU，可将该文件中的 `device` 改为 `cpu`、`adam_backend` 改为 `loop`。

各文件可独立改 `steps`、`learning_rate`、`adam_eps`、`replay_alpha`、`batch`、`gamma`、n-step/C51、设备/线程/回放/Adam、seed、生命规则和输出目录，不再强制三者相同。例如只改变 game_score 的 batch 和 learning_rate，不会影响另外两个文件。

先查看最终生效参数而不训练：

```bash
uv run contra-rl train --config configs/game_score.json --dry-run
```

优先级为：**显式 CLI 参数 > 配置文件 > 内置默认值**。布尔值也能双向覆盖，例如 `--all-lives` / `--no-all-lives`、`--standard-actor` / `--no-standard-actor`。一次临时短测可以覆盖预算和输出目录：

```bash
uv run contra-rl train --config configs/game_score.json \
  --steps 128 --warmup 32 --batch 8 --capacity 256 --max-steps 64 \
  --output runs/my-game-score-config-check
```

文件必须是平铺的 JSON 对象，键用下划线，与 CLI 的短横线选项对应（`learning_rate` ↔ `--learning-rate`）。`reward_mode`、`output` 必须明确给出。未知键、错误模式、错误类型、重复 JSON 键、NaN/Infinity、无效范围都会报错；配置的未知键/类型/模式错误不会被 CLI 覆盖掩盖。JSON 不支持注释；布尔值用 `true`/`false`，整数参数不能写成字符串或小数。

路径均相对于**运行命令的当前目录**，不是 JSON 所在目录，所以推荐在项目根目录运行。修改 seed 后请同时修改 output；已有 metrics.jsonl 的目录会拒绝覆盖。运行目录 config.json 记录最终合并后的参数、源配置绝对路径及 SHA-256；它包含运行元数据，与手写输入配置用途不同，不能原样作为输入配置。

| 参数 | 单位/含义 |
|---|---|
| `steps`、`max_steps`、`warmup` | agent 决策次数；分别为总预算、单 episode 上限、随机探索预热 |
| `train_every`、`target_every`、`checkpoint_every`、`log_every` | 每多少次 agent 决策触发；不是模拟器帧数 |
| `batch` | 每次学习采样的 transition 数 |
| `capacity` | 回放中的 n-step transition 容量 |
| `n_step`、`gamma` | n-step 的决策步数、每决策步折扣 |
| `learning_rate`、`adam_eps` | Adam 学习率和 epsilon |
| `replay_alpha` | PER 优先级指数，范围 [0,1] |
| `atoms`、`vmin`、`vmax` | C51 原子数与训练回报支持区间，按本组奖励单位解释 |
| `all_lives` | false 为首命结束，true 为允许复活直到 Game Over；均只研究第一关 |
| `seed`、`threads` | 随机种子、PyTorch CPU 线程数 |
| `profile`、`profile_sync` | 可选监测 JSON 路径（默认 null）、是否给诊断计时加设备同步 |

独立调参后，结果是“奖励 + 参数组合”的比较，**不能再声称实验只改变了奖励**。现有 `contra_rl.experiments` 与 `experiments/reward-comparison*.json` 继续作为统一共享参数的公平对比入口，不会读取或强制改动这三份独立配置。

本次仅做三组各 128 步/25 次更新的配置入口验证，详见 [独立配置验证记录](reports/independent-config-validation.json)。

## 已完成的性能优化

M4 实测、同样 2,048 步/449 次更新：CPU **70.3 → 78.0 步/s（+11%）**；MPS **64.4 → 两次平均 79.7 步/s（+24%）**。完整数据、阶段分解、CPU/GPU 采样、峰值内存和复现入口见 [性能报告](reports/performance/REPORT.md)。26 项 CPU/MPS 测试通过；CPU 对照训练后的全部模型张量逐位一致。

默认启用等价的因子化 NoisyNet 单帧推理。推荐的 Metal 参数为 `--device mps --threads 4 --replay-storage raw --adam-backend fused`；CPU 用 `--device cpu --threads 4 --replay-storage raw --adam-backend loop`。raw 以较多内存换取吞吐，满容量 20,000 的双份观测约 1.13 GB，默认仍为省内存的 zlib。`--standard-actor` 可还原物化噪声 actor。奖励、batch、采样/更新比例及结束规则不变。

```bash
# 三种奖励共用实测推荐的性能后端；默认只打印计划
uv run python -m contra_rl.experiments --config experiments/reward-comparison-fast-mps.json

# 复现短程性能检查，并保存 CPU/GPU/内存与阶段监测数据
uv run contra-rl train --device mps --threads 4 --replay-storage raw --adam-backend fused \
  --steps 2048 --warmup 256 --batch 32 --capacity 20000 --max-steps 1000 --seed 4 \
  --output runs/my-performance-check --profile runs/my-performance-check/profile.json
```

正常终端已验证 MPS 可用；受限执行沙箱可能报告 available=False，并不表示这台 M4 不支持 GPU。可用 `uv run python scripts/check_accelerator.py` 检查当前执行上下文。GPU 百分比来自系统全 GPU 计数器，包含其他应用；进程 CPU 百分比的 100% 表示一个核。`--profile-sync` 仅用于短程同步诊断，不用其屏障后的速度代表正常吞吐。fused Adam 的 FP32 舍入差异已对 float64 参考核验，详见报告。

## 三种训练奖励

| `--reward-mode` | 每个 agent step 的训练奖励 | 含义 |
|---|---|---|
| `score_clipped` | `sign(Δ游戏分数)` | 论文常用的 +1 / 0 / −1 裁剪方式 |
| `game_score` | `Δ游戏分数` | 真实游戏计分增量，不缩放、不裁剪 |
| `progress` | `clip(新历史最远摄像机进度/256 − 死亡次数, −1, 1)`；过关步覆盖为 +10 | 现有塑形设计，效果有待比较 |

这里的 Δ 是**变化量**，不是当前累计总分。一次动作最多重复 4 个模拟器帧：先逐帧计算并累加分数变化，再对总增量取 sign。例如 4 帧分别得 100、200、300、0 分，`score_clipped` 奖励为 **1**，`game_score` 为 **600**；不会逐帧取 sign 后加成 3。sign 在 n-step 回报累计之前应用。

`score_clipped` 参考 [DQN 论文第 5 节](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) 的训练奖励裁剪，只借用这一处理方式，不声称完整复现其 Atari 任务设定。初代 Contra 正常游戏不会因死亡扣分，因此该模式通常只有 0 / +1；不会为凑齐三种值而自行添加死亡 −1。−1 仅是 sign 对有效负分变化的定义。

### 计分读取和边界

`$07e2/$07e3` 是玩家 1 的小端二进制分数，**每 RAM 单位 = 画面 100 分**，不是 BCD。游戏绘制分数时在十进制数后补两个 0，见 [原游戏计分/绘制代码注释](https://github.com/vermiceli/nes-contra-us/blob/main/src/bank7.asm)。日志中 `score` 保留 RAM 原始单位，`game_score`、`game_score_delta` 和奖励 `game_score` 均使用真实分数单位。

- `reset()` 重新建立分数基线、清零 episode 统计，初始已有分数不当成新奖励；跨重置不求差。
- 游戏分数在 `0xffff` 饱和，不按 uint16 回绕；到上限后的真实计分增量为 0。遇到意外下降立即报错，防止把重置/错误 RAM 当成巨大负分或正向回绕奖励。
- 加命只记录 `extra_lives`；不当成死亡，不额外给奖励。原生模式只奖励伴随发生的原生分数变化。

旧模式 `native` 实际是 Stable Retro 的 **xscroll 滚屏奖励**，不是游戏计分，因此该名称已经移除，不设自动别名。滚屏奖励仍记录为 `integration_scroll_reward`。新环境版本是 `contra-level1-three-rewards-v2`；旧检查点、缺少明确奖励模式的配置都会被拒绝，不会静默套用新语义。旧训练产物保留作历史记录。

## 安装和单模式运行

```bash
uv sync --locked
uv run contra-rl doctor
uv run pytest -q
```

Python 3.12、项目 `.venv`、`uv.lock` 和项目内 `.uv-cache` 均由 uv 管理。现有依赖已安装，命令可用 `uv run --offline ...` 离线运行。

```bash
# 随机动作检查，保存 episode 汇总和逐步三奖励日志
uv run contra-rl probe --reward-mode game_score --seed 4 \
  --episodes 3 --max-steps 2000 --report runs/score-probe.json

# 单模式短测，其他两种替换 --reward-mode 即可
uv run contra-rl train --reward-mode score_clipped --seed 4 \
  --steps 1024 --warmup 512 --batch 16 --capacity 2048 \
  --max-steps 500 --vmin -10 --vmax 10000 --atoms 51 \
  --output runs/my-score-clipped-smoke

# 评估新版本短测检查点，明确匹配该检查点的奖励与生命规则
uv run contra-rl eval --reward-mode game_score \
  --checkpoint runs/reward-smoke-v2/seed-4/game_score/latest.pt \
  --episodes 3 --max-steps 500 --video runs/game-score-eval.mp4 \
  --report runs/game-score-eval.json
```

默认奖励仍是 `progress`。默认首命死亡就结束，`--all-lives` 允许复活直到最终 Game Over；**两种结束规则都独立于奖励模式**。所有模式在第一关 level 变化时结束，默认时间限制是 9,000 个决策步。不同奖励的比较必须固定生命规则、最大步数、动作、预处理、初始存档和评估方式。

## 可重复的三模式实验

[统一实验配置](experiments/reward-comparison.json) 显式固定三个 seed、每组预算、网络/采样/训练超参数和任务设置；每个 seed 下只改变 reward mode 和输出目录。运行器按顺序训练并评估三组，保存源码/锁文件 SHA-256、完整 manifest、各组配置、检查点和共同指标 CSV。

```bash
# 默认只打印全部命令，不启动训练、不创建输出目录
uv run python -m contra_rl.experiments

# 如需实际执行全部长期实验，显式添加 --run（目前没有替你执行）
uv run python -m contra_rl.experiments --run

# 已实际执行的三模式短程验证，可用新目录重现
uv run python -m contra_rl.experiments \
  --config experiments/reward-smoke.json --output-root runs/reward-smoke-repeat --run
```

长期配置为 seed 0/1/2、每模式每 seed 1,000,000 步，默认首命结束，所有组共用配置。可用 `--steps`、`--seeds`、`--output-root` 覆盖预算和路径，覆盖值会统一应用于三组并写入 manifest。需要改生命规则或超参数时编辑配置中的共享 `train` 设置，不能在某一模式下单独改。输出目录已有实验时拒绝覆盖。

### C51 的数值表示

旧 `[−10,10]` 区间会把原生 `+100/+1000` 目标大量投影到上界，因此新 CLI 默认、统一实验和短测配置均显式使用 **51 原子、[−10,10000]**，三组完全相同。通过 `--atoms --vmin --vmax` 可更改，但不会随 reward mode 自动改；评估从检查点恢复其支持区间。

这只是容纳较大分数的共同数值表示，**未证明是最佳区间**：宽区间的原子间距会降低小奖励的分辨率，长期原生回报仍可能超出上界。日志的 `c51_target_outside_mass` 是投影前落在区间外的目标概率质量；汇总有其训练平均值。`c51_projected_edge_mass` 记录投影后的边缘质量，但宽区间内的小奖励也可能落在边缘附近，不能单凭该指标断定饱和。正式结论应检查越界量，并以统一修改支持区间的敏感性实验确认稳定性。`game_score` 从不为适应区间而额外裁剪奖励。

## 日志和比较指标

每条 transition 日志同时包含：

- `game_score_delta`：当前 agent step 的真实分数增量；`game_score`：当前游戏总分。
- `integration_scroll_reward`：原 Stable Retro 滚屏奖励。
- `progress_reward`、`score_clipped_reward`：另外两种奖励的同轨迹数值。
- `training_reward`：本组实际送入回放和 Rainbow 的奖励。
- `progress_max`、`success`、`deaths`、`extra_lives`、生命值、实际推进帧数，以及各项 episode 累计量。

训练写入 `metrics.jsonl`：非零奖励/进度/加命和结束事件均记录，其余每 `--log-every` 步记录一次（默认 100，设 1 可全量记录）；每个 episode 和最后不足一个 episode 的部分都保留累计指标。`--report path.json` 的评估/随机检查另外写入 `path.steps.jsonl`，逐步记录全部奖励向量。

**跨模式比较成功率、真实得分增量、最远摄像机进度、死亡/加命次数和决策/模拟器帧数。不要用各模式的训练 return 大小直接排名。** `comparison.csv` 只汇总这些共同任务指标和 C51 诊断，不将奖励尺度不同的 return 当成能力指标。确定性固定存档加 greedy 评估可能重复完全相同的轨迹，重复 episode 不是独立泛化样本；正式研究还需要多个训练 seed，后续若增加起始随机化也应三组统一。

检查点包含网络、优化器、步数、完整配置、动作映射、环境版本和 C51 支持参数；回放/模拟器/随机数状态未保存，不支持精确断点续训。评估严格检查环境版本、奖励模式和生命规则。

## 共同环境与实现

- 25 个组合动作：方向/斜向瞄准、B 射击、A 跳跃，排除 START/SELECT 和相反方向。
- 重复动作 4 帧，最后两帧最大值，灰度 84×84，4 帧堆叠；固定 Level1，无额外随机 reset no-op。
- RAM 修复独立于奖励：角色状态 `$90`，最终 Game Over `$38`；`lives=0` 仍有最后一条可玩的命。
- `progress` 的摄像机进度来自 `$64/$65`，不是玩家屏幕 X。只计算当前关、每模拟器帧增量 ≤16、范围 ≤8192 的新最大值；死亡 −1；普通步裁剪，过关步覆盖 +10。它是现有待比较设计，不能预先认定比其他两种更好。
- 时间限制使用 truncated、保留 bootstrap；死亡/过关 terminated 不 bootstrap。n-step 清队列，不能跨 reset。
- Rainbow：Double DQN、PER、Dueling、n-step、C51、NoisyNet；训练探索用 NoisyNet、warmup 随机动作，评估关闭噪声。实现选择/许可证核查见 [实现评估](reports/implementation-review.md)。

默认 CPU、4 线程；正常终端 MPS 已实测可用，受限沙箱可能误报不可用，`doctor` 和 `scripts/check_accelerator.py` 可检查当前上下文。回放容量默认 20,000，zlib/raw 均可配置；未压缩双份观测约 1.13 GB，另有模型开销。步数均指 agent 决策，最多对应 4 倍模拟器帧。

## ROM 与验证产物

仓库包含 `roms/Contra-USA.nes`，131,088 字节。去 16 字节 iNES 头的 SHA-1 为 `979494e7869ac7ab4815fdbd1dc99f893f713fbf`，与 Stable Retro 集成匹配。来源、访问及核查记录见 [ROM 下载记录](reports/rom-provenance.json)。克隆到 Colab 或其他新环境并安装依赖后，需要将仓库里的 ROM 导入 Stable Retro：

```bash
uv run contra-rl import-rom roms/Contra-USA.nes
uv run contra-rl doctor
```

- [三奖励真实 ROM 验证](reports/three-reward-rom-validation.json)：每组重放同一 4,000 动作序列，物理观测/共同指标完全相同，并观察到 +100 游戏分数对应 +1 裁剪奖励。
- [短程实验配置](experiments/reward-smoke.json)、`runs/reward-smoke-v2/manifest.json`、`comparison.csv`：每模式同 seed、同预算的训练/评估执行验证，不是效果结论。
- `uv run python scripts/validate_rewards.py`：重现相同动作下的三奖励数值与轨迹一致性检查。
- `uv run python scripts/validate_contra.py`：重置、首命结束、三次死亡及最后 Game Over 检查。
- `tests/`：分数单位/二进制进位/饱和/重置/加命、先累加再 sign、三模式共同任务、旧语义拒绝、统一实验参数，以及原有 Rainbow 核心测试。

`runs/contra-validated/`、旧 `reports/contra-*` 是 v1 历史结果；新 CLI 不会自动加载其中的旧检查点。实际打通第一关仍未验证，过关奖励分支由针对性测试覆盖。
