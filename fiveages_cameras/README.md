# fiveages_cameras

本目录为相机抽象与厂商 SDK 封装（如奥比 Orbbec 等），**不在本公开仓库中提供源码**。

使用 `tools/orbbec_tracking_live.py` 前，请将贵司或厂商提供的相机模块代码放到本目录，并保持脚本期望的包结构（例如 `orbbec/orbbec_camera_usb.py` 与相关 YAML）。

若你只有本仓库而没有相机子项目，可先仅运行 `tools/demo_tracking.py` 等对本地视频的示例。
