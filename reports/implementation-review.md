# 实现选择与资料核查（2026-10-20）

实际检查了 [Kaixhin/Rainbow](https://github.com/Kaixhin/Rainbow)：README 声明完整六项 Rainbow 扩展，MIT 许可证；GitHub commits API 返回最后提交为 2022-01-13，SHA `1745b184c3dfc03d4ffa3ce2342ced9996b39a60`。requirements.txt 仍列出 `atari-py`、OpenCV、Plotly、torch、tqdm，不直接支持当前 Gymnasium + Stable Retro NES。它是有价值的参考，但截至核查没有近期维护记录，不作为此项目安装依赖。

这里独立实现小型 PyTorch Rainbow，未复制或 vendoring 该仓库代码：

| 六项扩展 | 实现位置 | 关键行为 |
|---|---|---|
| Double DQN | `agent.py` | 在线网络选下一动作，目标网络提供该动作的分布 |
| Prioritized replay | `replay.py` | sum/min tree，比例采样，重要性权重，分布 KL 优先级 |
| Dueling | `model.py` | value + advantage − mean advantage，在 logits 层合并 |
| n-step | `replay.py` | 默认 3 步；episode 边界清队列；时间截断保留 bootstrap |
| C51 | `model.py` | 51 原子 categorical projection，整数落点保留全部概率质量 |
| NoisyNet | `model.py` | factorised Gaussian；训练重采样，评估使用均值 |

依据：[Rainbow 论文](https://arxiv.org/abs/1710.02298)、[Stable Retro 入门](https://stable-retro.farama.org/getting_started/)、[macOS 文档](https://stable-retro.farama.org/macos_installation/)。本地实验只验证正确执行，尚未复现论文 Atari 成绩，也没有证明通关能力。

## Contra RAM 修正

交叉核查公开的 [美版 Contra 反汇编 RAM 注释](https://github.com/vermiceli/nes-contra-us/blob/main/src/ram.asm) 和 [bank7 计分代码](https://github.com/vermiceli/nes-contra-us/blob/main/src/bank7.asm)：

- 上游 experimental/data.json 的 player_state=326 (`$0146`) 位于音频内存区，真正 PLAYER_STATE 是 `$0090`=144：0 初始下落、1 正常、2 死亡、3 不能移动。
- `$0032` 的 lives=0 表示最后一条可玩的命；最终游戏结束看 `$0038` 的 P1_GAME_OVER_STATUS=1。
- `$0064` 是关卡内屏幕编号，`$0065` 是该屏幕滚动像素偏移，组合为 big endian 16 位摄像机进度；不是玩家屏幕 X。
- `$07e2/$07e3` 为小端二进制计分，每单位对应画面 100 分；progress 模式不使用计分；v2 另外提供 score_clipped 与 game_score 模式，作为待比较基线。

本地 integration/data.json 和 scenario.json 覆盖错误字段，不改动 site-packages 的集成文件。实际第一关、右行射击脚本验证了正常状态→死亡→复活，命数 2→1→0→Game Over；见 contra-all-lives.json。过关奖励条件使用 level 变化，仅单元测试覆盖，尚未实际打通第一关验证。

## 与错误修复分开的设计选择

默认 progress 奖励（摄像机进度高水位 /256、死亡 −1、过关 +10）和默认首命结束都是额外研究设定，不是 Rainbow 的必需组件，也不属于 RAM 错误修复。`--reward-mode native` 返回原始集成滚动奖励，不缩放、不裁剪、不另加死亡/过关奖励；`--all-lives` 允许复活。两种奖励模式共享已修正的 RAM 数据。


## v2 三奖励语义与比较

用户随后要求三种奖励对比，v2 已移除歧义的 native 名称。score_clipped 对一次 agent step 的游戏分数增量总和取 sign；game_score 使用同一增量的真实点数（RAM ×100）；progress 保留已有摄像机高水位塑形。原始 xscroll 集成奖励只作诊断。三种奖励不改变生命/结束规则，比较配置共用动作、存档、预处理和全部训练超参数。

DQN 裁剪依据：[论文第5节](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)。计分绘制补两个 0、uint16 饱和与加命逻辑依据：[游戏 bank7 代码](https://github.com/vermiceli/nes-contra-us/blob/main/src/bank7.asm)。游戏分数正常不随死亡下降；reset 单独设基线，意外下降报错，不将其编造成死亡负奖励。

CLI/实验默认共用 C51 的 51 原子、[−10,10000] 区间，显式记录目标投影前越界质量，避免静默把原生 +100/+1000 挤到旧 [−10,10] 上界；原生奖励自身不裁剪。此共享区间未调优，小奖励分辨率和长期大回报越界仍需用共同参数敏感性实验检查。旧环境版本检查点被显式拒绝，不自动迁移奖励语义。
