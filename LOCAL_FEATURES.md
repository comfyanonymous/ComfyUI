# ComfyUI-Inference-Scaling 本地新功能说明

> 基于 ComfyUI 官方仓库，本项目在 `feature/inference-scaling` 分支上添加了以下自定义功能。

---

## 新增功能概览

### 1. 推理缩放节点 (Inference Scaling Nodes)

**文件：** `comfy_api_nodes/nodes_inference_scaling.py`

在标准 KSampler 采样流程中加入了**实时图像质量监控**机制，可以在生成过程中（而非仅在最终结果）动态评估图像质量。

#### 1.1 QualityVerifier（质量验证器基类）

- 定义了图像质量评估的抽象接口
- 输入：`torch.Tensor` 格式的图像（`[B, H, W, C]`，范围 `[0, 1]`）
- 输出：`(is_acceptable: bool, quality_score: float)` 元组

#### 1.2 SimpleVerifier（简单质量验证器）

基于以下两项指标评估图像质量：

| 指标 | 方法 | 权重 |
|------|------|------|
| 图像方差 | 高方差 = 更多细节 | 60% |
| 边缘强度 | 类 Sobel 算子检测结构 | 40% |

质量分数归一化到 `[0.0, 1.0]`。

#### 1.3 VerifierSelectionNode（验证器选择节点）

ComfyUI 节点，用于创建和配置质量验证器。

**输入参数：**
- `verifier_type`：验证器类型（`"simple"` / `"custom"`）
- `quality_threshold`：最低质量分数阈值（`0.0–1.0`，默认 `0.5`）

**输出：**
- `verifier_id`：验证器的字符串 ID，供 InferenceScalingNode 使用

#### 1.4 InferenceScalingNode（推理缩放节点）

核心节点，将标准采样流程包装并加入质量监控回调。

**输入参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `verifier_id` | String | `""` | 来自 VerifierSelectionNode 的验证器 ID |
| `model` | Model | — | 扩散模型 |
| `positive` | Conditioning | — | 正向提示词条件 |
| `negative` | Conditioning | — | 负向提示词条件 |
| `latent_image` | Latent | — | 初始潜空间图像 |
| `vae` | VAE | — | 用于步骤中间解码的 VAE 模型 |
| `seed` | Int | `0` | 随机种子 |
| `steps` | Int | `20` | 采样步数 |
| `cfg` | Float | `7.0` | CFG 引导系数 |
| `sampler_name` | Combo | `"euler"` | 采样器名称 |
| `scheduler` | Combo | `"normal"` | 调度器名称 |
| `check_interval` | Int | `5` | 每 N 步检查一次质量 |
| `quality_threshold` | Float | `0.5` | 质量调整触发阈值 |
| `scale_factor` | Float | `1.2` | 质量低时的参数缩放系数 |

**输出：**
- `latent`：去噪后的潜空间图像（格式与标准 KSampler 一致）

**工作原理：**
1. 在每个采样步骤的回调中，按 `check_interval` 间隔解码当前潜空间
2. 使用 VAE 将潜变量解码为图像（`[B, H, W, C]`）
3. 调用验证器评估质量分数
4. 记录质量历史，低质量时输出警告日志
5. 采样完成后返回标准格式的 latent dict

---

### 2. 节点开发教程

**文件：** `HOW_TO_WRITE_NODES.md`

详细说明了如何基于本项目的 ComfyAPI 系统编写自定义节点，包括：

- `IO.ComfyNode` 的继承结构
- `define_schema()` 方法的编写方式（输入/输出定义）
- `execute()` 异步方法的实现模式
- 通过 `ComfyExtension` 注册节点的方式
- 各种输入类型的使用示例（String、Int、Float、Image、Combo 等）

---

### 3. 节点模板示例

**文件：** `comfy_api_nodes/nodes_template_example.py`

可直接复制修改的节点模板，演示了：

- 文本输入（`IO.String.Input`）
- 数值输入带滑块（`IO.Int.Input` + `IO.NumberDisplay.slider`）
- 可选图像输入（`IO.Image.Input`, `optional=True`）
- 多类型输出
- 完整的 `execute()` 实现示例

---

### 4. 调试笔记

**文件：** `DEBUGGING_NOTES.md`

记录了开发过程中发现并修复的问题：

| 问题 | 状态 | 解决方案 |
|------|------|----------|
| 噪声/潜变量处理错误 | ✅ 已修复 | 正确使用 `comfy.sample.prepare_noise()` 生成噪声 |
| KSampler 实例化方式错误 | ✅ 已修复 | 改用 `comfy.sample.sample()` 直接调用 |
| VAE 解码张量格式问题 | ✅ 已修复 | 正确处理 `[B,C,H,W]` → `[B,H,W,C]` 转换 |
| 回调闭包变量作用域 | ✅ 已修复 | 添加 `nonlocal` 声明 |
| 错误处理不完善 | ✅ 已改进 | 质量检查失败不中断采样，仅记录警告 |
| latent dict 键丢失 | ✅ 已修复 | 使用 `latent_image.copy()` 保留所有键 |

---

## 已知限制

- **CFG 动态调整：** 由于 CFG 在采样开始时确定，无法在采样过程中实时修改
- **VAE 解码开销：** 每 N 步解码一次会增加额外计算时间（可通过 `check_interval` 控制频率）

## 未来改进方向

- 实现早停机制（质量持续良好时提前结束采样）
- 支持基于质量的动态步数调整
- 引入更高级的质量评估方法（如感知损失、CLIP 评分）
- 使用 TAESD 替代完整 VAE 加速预览解码

---

## 使用方式

```
1. 添加 VerifierSelectionNode
   - verifier_type: "simple"
   - quality_threshold: 0.5

2. 添加 InferenceScalingNode
   - 连接 model / positive / negative / latent_image / vae
   - 连接 verifier_id（来自 VerifierSelectionNode）
   - steps: 20, cfg: 7.0
   - check_interval: 5（每5步检查一次）
   - scale_factor: 1.2

3. 连接后续的 VAE Decode 节点查看最终图像
```

---

## 分支与版本信息

- **自定义分支：** `feature/inference-scaling`
- **基于 ComfyUI 版本：** 已 rebase 到上游最新提交（`f21f6b22`，2026-04-04 更新）
- **自定义提交：** `c254fa10 Add inference scaling nodes with quality monitoring during generation`
