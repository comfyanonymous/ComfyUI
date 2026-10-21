# H3 本地依赖环境

本机为 RTX 5060 8GB、32GB 内存。此配置补齐 H3 主路径依赖，不表示原尺寸长视频已通过显存或质量验收。

## 启动与持久化

`docker-compose.override.yml` 自动叠加到原配置，使用 `comfyui-sos-comfyui:h3-local`。五个数据挂载、GPU 设置、`--disable-cuda-malloc` 和 1.5GB 显存预留保持不变，只显式补上管理器启用标志。

```bash
docker compose config --quiet
docker compose up -d --no-deps --no-build comfyui
```

重建派生镜像需要本机 `temp/h3-build/wheels/` 中的 SageAttention 与 soundfile 两个固定 wheel，权重和 wheel 不提交 Git。构建前同步受版本管理的缺包锁文件到小构建上下文：

```bash
cp h3_local_requirements.txt temp/h3-build/wheels/h3_local_requirements.txt
cp h3_legacy_requirements.txt temp/h3-build/wheels/h3_legacy_requirements.txt
docker build -f Dockerfile.h3-local -t comfyui-sos-comfyui:h3-local temp/h3-build/wheels
```

`Dockerfile.h3-local` 校验 wheel 摘要并使用 `--no-deps` 安装，禁止顺带升级或降级 PyTorch。Fill 还会整体导入其音频、SDK、PDF 等节点，因此将独立环境实际验证所需的 19 项缺包以版本和 wheel 摘要固定在 `h3_local_requirements.txt`，安装启用 `--require-hashes`，不全量安装扩展的浮动 requirements。它基于本机已有、验收过的 `vision-fixed` 镜像，不是适用于任意设备的通用安装包。

## 固定版本

| 组件 | 本次版本或提交 |
| --- | --- |
| PyTorch / CUDA | `2.11.0+cu128`，未修改 |
| ComfyUI | `0.33.0`，未修改核心 |
| 管理器 | 内置 `comfyui_manager==4.2.2`，不另装第二份 |
| T8 采样扩展 | 原作者 v1.87.0：`T8mars/comfyui-minimax-h3-audio-T8@4af2a0256eb5dccc0dabc6256de6b9527bfd8531` |
| SageAttention | 官方 `thu-ml/SageAttention@d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`，版本 `2.2.0` |
| Sage wheel SHA256 | `93ecec1aeee32f4f58f7d9b908f8b442b8dea678f9f5ab935ae3bc288d0658bb` |
| Fill 音频导入依赖 | `soundfile==0.14.0`，以固定 wheel 和 `--no-deps` 补齐 |
| Derfuu 旧转换节点 | `Derfuu/Derfuu_ComfyUI_ModdedNodes@d0905bed31249f2bd0814c67585cf4fe3c77c015`，v1.0.1 |
| Impact Pack 旧开关节点 | `ltdrdata/ComfyUI-Impact-Pack@429d0159ad429e64d2b3916e6e7be9c22d025c3c`，v8.28.3 |
| Impact 最小新增依赖 | `piexif==1.1.3`、`dill==0.4.1`，固定 wheel 摘要，不解析传递依赖 |

管理器配置保持 `security_level=normal`、安全扫描及 SSL 校验，下载渠道改为 `https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/main`。将官方支持的 `use_unified_resolver` 设为 `False`，避免每次重启都无条件重算并安装全部旧扩展依赖；界面内明确发起的安装仍可使用正常逐扩展流程。原 `custom_nodes/ComfyUI-Manager` 目录不删除，内置模式会阻止重复加载。

T8 与 Sage 原作未达一万星，但目标工作流需要其专用接口，没有已核验的一万星等价替代；采用固定原作，不伪造同名节点或替换成不等价 LoRA。

T8 最新版的附加保存节点依赖 `Hidden.execution_list`，当前运行镜像没有此新 API。固定 v1.87.0，而非接受部分注册或修改核心；已确认其 `h3_t8/nodes.py`、`sampling.py` 和 `core.py` 与最新版逐字一致，目标采样器和 H3 核心计算不变。

v1.87.0 全部 373 个节点的结构定义在实际镜像下通过检查。独立真实 ComfyUI 服务同时加载 Fill 与 T8，拼图节点和双时钟采样器均在 `/object_info` 中，本轮无扩展导入失败。

## 模型清单

四个主权重来自公开仓库 `Comfy-Org/MiniMax-H3@e5eb578a89295337b8ff433a035929ce0279e0b6`：

- `models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors`
- `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `models/vae/minimax_h3_video_vae_fp16.safetensors`
- `models/vae/minimax_h3_audio_vae_fp32.safetensors`

另外两个专用权重：

- `models/loras/minimax_h3_turbo_4步加速_comfyui_T8.safetensors`，来自 `RunningHubAI/rh-minimax-h3-turbo-4-comfyui-t8-lora@77c381d0670b1cfbc1f98e1a5eab99689e3189da`。
- `models/RMBG/BiRefNet/BiRefNet_toonout.safetensors`，来自 `1038lab/BiRefNet@4d000788a9698c7f8d67c8c6ce2b40c768f5b909`。

合计 57,203,835,765 字节。下载保留断点，完成后逐文件核对固定发布摘要、大小和 safetensors 头；未完成文件不能作为可用模型。

2026-10-21：上述 6 个文件已全部完成大小、SHA256 及安全张量头校验；未加载这些权重运行完整 H3。

## 构建和验收边界

构建发生在独立容器：GCC 12、NVIDIA 官方 CUDA 12.8.93 编译器、`TORCH_CUDA_ARCH_LIST=12.0`、单编译任务。生产容器不安装编译器，不修改原作源码。

Debian 13 的 glibc 2.41 与 CUDA 12.8 头文件有声明冲突，复用 [Sunshine 固定版本的兼容补丁](https://github.com/LizardByte/Sunshine/blob/3494b23e20b13d27b88fa08b97d07cb6d917f919/packaging/linux/patches/x86_64/cuda-12-math_functions.patch)。只作用于临时工具链，断言恰好修改 12 条声明，不改生产系统头文件、ComfyUI 或注意力算法。补丁 SHA256 为 `a3a0e5c1f608b18b1269fea51ac34ca555ad4dba32d8c32877831271602a2962`。

候选镜像在 RTX 5060 上通过 48 项小张量测试：FP16/BF16、1024/1031 长度、两个随机种子、连续/融合 QKV 非连续布局，以及 KJ 实际 H3 注意力函数的分头路由。相对 FP32 SDPA 的最大相对 L2 误差为 0.037675，最小余弦相似度为 0.999290，测试峰值显存为 222,526,464 字节。

测试采用 eager（即时执行），不使用 CUDA Graph；未加载 H3 模型，也未验证真实权重、RoPE、长视频显存峰值或成片质量。当前 CUDA 12.8 对 Comfy Kitchen 优化内核的限制仍存在，不能把注意力验收理解为整个生成链已通过。

本机下载、构建、验收脚本及完整结果保存在 `/opt/maoning/Reference/h3-local-setup-20261021/`；它们不进入 ComfyUI 核心或 Git 中的模型目录。

H3 主路径依赖验收镜像为 `50460a79a4f1f7552432ba26b4f4675c65c89fba33c31bd9a6743f84e6522e73`，补齐旧分支前另保留为 `comfyui-sos-comfyui:before-legacy-20261021`。该阶段重启后扩展导入失败数为 0，T8 实际注册 373 个节点；H3 模型、编码器、VAE、LoRA 和 toonout 均出现在相应选项中，所有活跃后端节点齐全。管理器版本接口与临时 Chrome 页面检查通过，48 项 GPU 验收在该镜像再次通过。

与原基础镜像比对，原有 241 个 Python 包无版本变更或移除；新增 21 项固定依赖。另外既有扩展自身启动时补入 8 项小依赖（Socket.IO 相关 5 项及 jsonpointer、jsonpatch、lark），没有变更已有包。关闭的是管理器的无条件批量重算，不是修改这些扩展的自有启动行为。

原工作流保持不变，SHA256 为 `ea78e0e23d665a31f34e4c762a84524248b20b846df9dfeb3b138621335baa41`。其 11 个 `DF_Int_to_Float` 和一个 `ImpactSwitch` 全处于旁路且没有连向输出，不是 H3 主路径依赖；随后用户明确要求保留旧分支并尽量补齐全部缺失节点，因此按下节安装原作，未修改画布或伪造同名节点。8 个空参考槽、节点 231/268 引用的本机不存在图片，仍需在真正试跑前整理；依赖齐全不表示原图可以不改素材直接排队。

## 补齐旧分支

两个原作均低于一万星，Derfuu 在 2024-11 归档；作为用户明确选择的旧图兼容例外，不将其当作活跃的新项目选型。两者源码直接从原作者 GitHub 克隆并固定上表提交，不修改原作实现。

下列命令仅适用于两个目标目录尚不存在的安装环境；已有目录应先检查本地改动，不覆盖：

```bash
git clone --filter=blob:none https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes.git custom_nodes/Derfuu_ComfyUI_ModdedNodes
git -C custom_nodes/Derfuu_ComfyUI_ModdedNodes switch --detach d0905bed31249f2bd0814c67585cf4fe3c77c015
git clone --filter=blob:none https://github.com/ltdrdata/ComfyUI-Impact-Pack.git custom_nodes/ComfyUI-Impact-Pack
git -C custom_nodes/ComfyUI-Impact-Pack switch --detach 429d0159ad429e64d2b3916e6e7be9c22d025c3c
```

Derfuu 使用现有 NumPy 和 ComfyUI，自身没有安装脚本或新依赖。Impact 完整包硬导入所需的 segment-anything、scikit-image、SciPy、OpenCV、Transformers 等已存在，镜像仅通过 `h3_legacy_requirements.txt` 补充 piexif 和 dill；不再装一份会覆盖现有 OpenCV 的 headless 发行包，不安装整份无上限 requirements。

本次不运行 Impact 的 `install.py`，避免其自动下载 SAM ViT-B、生成配置及清理旧目录；正常原作导入无需此脚本。SAM2 在源码中明确为可选依赖，这次未安装，也未下载其权重。开关节点验收通过不等于 SAM2 分割功能已验收。

独立真实 CPU ComfyUI 注册 Derfuu 全部 31 个、Impact 全部 197 个节点。通过 `/prompt` 实际执行 17 项无模型图：15 个整数值转 FLOAT（含旧图 5–15）及两路 lazy 开关，返回数值、分支标签和索引均符合预期。过程未执行 H3、未生成视频、未写用户输入或模型。检查脚本、安装前基线和结果位于 `/opt/maoning/Reference/h3-legacy-setup-20261021/`。

2026-10-21 正式服务复验：当前镜像 `a9796897817f2ef121062bc9dba942e18135a163fe17a1833ee842f7b67297cf`，共注册 3874 个节点，原 3646 个无移除。目标工作流 104 个节点涉及的 49 种后端类型全部可用，包含旁路分支，缺失数为 0；前端虚拟节点不计入后端。原 270 个 Python 包版本全部未变，仅新增上述两个包。17 项实际执行再次全通过，工作流摘要不变，T8 373 个节点、六个模型文件和选项、管理器版本接口、两原作前端脚本均通过检查。重启日志没有扩展导入失败或这些测试的执行错误；不表示其他扩展可选功能或完整 H3 生成已通过测试。

## 恢复原服务

仅在队列为空时操作。显式指定原 Compose 文件即可不叠加 H3 配置：

```bash
docker compose -f docker-compose.yml up -d --no-deps --no-build comfyui
```

原镜像保留为 `comfyui-sos-comfyui:vision-fixed` 和 `comfyui-sos-comfyui:before-h3-20261021`，镜像 ID 为 `ba67d5aec57caf246473bc7c41ac05ad647e903775d2f05f057aac36fd489277`。管理器原配置备份在 `user/__manager/config.ini.before-h3-20261021.bak`，恢复前先检查是否已有用户后续修改。
