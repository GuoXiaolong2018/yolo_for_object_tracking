# vendor_cameras

本目录用于放置**第三方或自研**的相机抽象与厂商 SDK 封装（如奥比 Orbbec 等），**不在本公开仓库中提供具体实现**。

使用 `tools/orbbec_tracking_live.py` 前，请将相机相关代码放到本目录，并保持脚本期望的包结构（例如 `orbbec/orbbec_camera_usb.py` 与相关 YAML）。若相机代码里曾使用旧版日志包名，请将其改为与本仓库脚本一致的 `vendor_cam_utils` 导入方式。

若你只有本仓库而没有相机代码包，可先仅运行 `tools/demo_tracking.py` 等对本地视频的示例。
