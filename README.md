# YOLO 多目标跟踪（FiveAges）

基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 与 ByteTrack 的多目标跟踪示例，并提供奥比中光 USB 相机实时取流与 YOLO 跟踪、画面可视化的脚本。

## 目录结构

| 路径 | 说明 |
|------|------|
| `tools/demo_tracking.py` | 对本地视频文件做 `model.track(...)` 的极简示例 |
| `tools/orbbec_tracking_live.py` | 奥比相机实时 RGB 流 + YOLO 跟踪与可视化 |
| `tools/bytetrack.yaml` | ByteTrack 跟踪器参数（供 `track(..., tracker=...)` 使用） |
| `fiveages_cameras/` | 相机抽象目录（Orbbec 等）；公开仓库不包含内网子模块，请自备相机相关代码 |

## 环境要求

- Python 3.10+（建议 3.10 或 3.11）
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda
- Git

## 获取代码

```bash
git clone <本仓库 URL>
cd <仓库根目录>
```

将相机相关代码按 `fiveages_cameras/README.md` 说明放入 `fiveages_cameras/` 后再运行实时相机脚本。

## 安装依赖（Conda）

```bash
conda create -n yolo-tracking python=3.11 -y
conda activate yolo-tracking
pip install -U pip
pip install -r requirements.txt
```

需要 **GPU 加速** 时，建议先在**当前已激活的 conda 环境**中，按 [PyTorch 官网](https://pytorch.org) 选择 CUDA 版本并安装 `pytorch`、`torchvision`（官网提供 conda 或 pip 命令），再执行 `pip install -r requirements.txt`，避免默认装成仅 CPU 的 PyTorch。

**奥比相机实时脚本**（`orbbec_tracking_live.py`）另需安装厂商提供的 **pyorbbecsdk**（常见为 SDK 内 wheel 或内部 pip 源），不在本 `requirements.txt` 中固定版本。

## 模型权重

`.pt` 权重文件体积较大，通常不纳入版本库。请自备检测模型，或在脚本中改为官方权重名（如 `yolo11n.pt`），由 Ultralytics 自动下载。

## 运行示例

在**项目根目录**执行（保证 `fiveages_cameras` 存在且子模块已初始化）。

### 1. 视频文件跟踪（`demo_tracking.py`）

编辑 `tools/demo_tracking.py` 中的模型路径与视频路径后：

```bash
cd tools
python demo_tracking.py
```

脚本中默认使用同级目录下的 `bytetrack.yaml` 作为跟踪配置。

### 2. 奥比 USB 相机实时跟踪（`orbbec_tracking_live.py`）

```bash
cd <仓库根目录>
python tools/orbbec_tracking_live.py \
  --model /你的路径/权重.pt \
  --tracker tools/bytetrack.yaml
```

常用参数：

- `--config`：奥比相机 YAML，默认指向 `fiveages_cameras/orbbec/orbbec_camera_usb_config.yaml`
- `--serial`：覆盖配置中的设备序列号
- `--device`：如 `cuda:0` 或 `cpu`

预览窗口下按 `q` 可退出。界面中文依赖系统字体；若显示为方框，可安装 Noto CJK 等字体，或使用 `--font` / 环境变量 `ORBBEC_REC_FONT` 指定字体路径。

## 相机说明

`fiveages_cameras` 的接口与配置需由你方提供的相机代码包补齐；占位说明见 `fiveages_cameras/README.md`。

