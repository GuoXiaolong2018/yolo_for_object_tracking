#!/usr/bin/env python3
"""
奥比中光 USB 相机实时取流 + Ultralytics YOLO 多目标跟踪（persist=True 连续帧关联）。

复用 orbbec_record_video 的 fiveages_cam 导入与 BGR 转换逻辑；跟踪 API 参考 demo_tracking.py
与 material.txt 中的「Streaming for-loop with tracking」示例。

录制（与 orbbec_record_video 交互习惯一致，保存内容为当前窗口中的可视化画面，含检测/分割/轨迹/推理耗时条）：
  r     开始录制
  s     结束并保存，文件名：YYYYMMDD_HHMMSS_录制秒数.mp4
  空格  取消本段（不保存临时文件）
  q     退出（录制中则丢弃当前段）

依赖：pyorbbecsdk、opencv-python、numpy、ultralytics、torch、Pillow（中文界面）等。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import time
import types
from pathlib import Path


def _setup_fiveages_cam_imports(project_root: Path) -> None:
    cam_root = project_root / "fiveages_cameras"
    if not cam_root.is_dir():
        raise RuntimeError(f"未找到 fiveages_cameras 目录: {cam_root}")

    sys.path.insert(0, str(cam_root))

    logger_path = cam_root / "utils_backup" / "logger.py"
    if not logger_path.is_file():
        raise RuntimeError(f"未找到备用 logger: {logger_path}")

    pkg = types.ModuleType("fiveages_utils")
    pkg.__path__ = []
    sys.modules["fiveages_utils"] = pkg

    spec = importlib.util.spec_from_file_location("fiveages_utils.logger", logger_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 fiveages_utils.logger")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fiveages_utils.logger"] = mod
    spec.loader.exec_module(mod)


def _packet_to_bgr(packet, width: int, height: int, cv2) -> object | None:
    rgb = packet.get("rgb")
    if rgb is None:
        return None
    if rgb.shape[0] != height or rgb.shape[1] != width:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _make_save_filename(end_time: float, duration_sec: float) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(end_time))
    dur = f"{duration_sec:.2f}"
    return f"{stamp}_{dur}.mp4"


def _discard_recording(writer, temp_path: str | None, cv2) -> None:
    if writer is not None:
        writer.release()
    if temp_path and os.path.isfile(temp_path):
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _cjk_font_candidates(user_font: Path | None) -> list[Path]:
    paths: list[Path] = []
    if user_font is not None:
        p = user_font.expanduser().resolve()
        if p.is_file():
            paths.append(p)
    env = os.environ.get("ORBBEC_REC_FONT")
    if env:
        ep = Path(env).expanduser()
        if ep.is_file():
            paths.append(ep)
    paths.extend(
        [
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
            Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
            Path.home() / ".local/share/fonts/NotoSansCJK-Regular.ttc",
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
        ]
    )
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        k = str(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


class _PILTextOverlay:
    """与 orbbec_record_video 一致：Pillow 绘制中文。"""

    def __init__(self, cv2_mod, np, user_font: Path | None):
        self._cv2 = cv2_mod
        self._np = np
        self._font_path: Path | None = None
        for cand in _cjk_font_candidates(user_font):
            if cand.is_file():
                self._font_path = cand
                break
        self._cache: dict[int, object] = {}
        self._warned_no_cjk = False

    def _get_font(self, size_px: int):
        from PIL import ImageFont

        if size_px in self._cache:
            return self._cache[size_px]
        font = None
        if self._font_path is not None:
            try:
                font = ImageFont.truetype(str(self._font_path), size_px)
            except OSError:
                try:
                    font = ImageFont.truetype(str(self._font_path), size_px, index=0)
                except OSError:
                    font = None
        if font is None:
            if not self._warned_no_cjk:
                print(
                    "警告: 未找到可用的中文字体，界面中文可能显示为方框。"
                    " 请安装 fonts-noto-cjk 或使用 --font / 环境变量 ORBBEC_REC_FONT。",
                    file=sys.stderr,
                )
                self._warned_no_cjk = True
            font = ImageFont.load_default()
        self._cache[size_px] = font
        return font

    def draw(
        self,
        bgr,
        text: str,
        org_xy: tuple[int, int],
        font_px: int,
        color_bgr: tuple[int, int, int],
        *,
        stroke: int = 0,
        stroke_bgr: tuple[int, int, int] | None = None,
    ) -> None:
        from PIL import Image, ImageDraw

        x, y = org_xy
        font = self._get_font(font_px)
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        fill = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        if stroke > 0 and stroke_bgr is not None:
            sf = (int(stroke_bgr[2]), int(stroke_bgr[1]), int(stroke_bgr[0]))
            dr.text((x, y), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=sf)
        else:
            dr.text((x, y), text, font=font, fill=fill)
        bgr[:, :, :] = self._cv2.cvtColor(self._np.asarray(im), self._cv2.COLOR_RGB2BGR)

    def text_size(self, text: str, font_px: int) -> tuple[int, int]:
        from PIL import Image, ImageDraw

        font = self._get_font(font_px)
        dr = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = dr.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def draw_centered(
        self,
        bgr,
        text: str,
        center_xy: tuple[int, int],
        font_px: int,
        color_bgr: tuple[int, int, int],
        *,
        stroke: int = 0,
        stroke_bgr: tuple[int, int, int] | None = None,
    ) -> None:
        from PIL import Image, ImageDraw

        cx, cy = center_xy
        font = self._get_font(font_px)
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        bbox = dr.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = int(cx - tw / 2)
        y = int(cy - th / 2)
        fill = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        if stroke > 0 and stroke_bgr is not None:
            sf = (int(stroke_bgr[2]), int(stroke_bgr[1]), int(stroke_bgr[0]))
            dr.text((x, y), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=sf)
        else:
            dr.text((x, y), text, font=font, fill=fill)
        bgr[:, :, :] = self._cv2.cvtColor(self._np.asarray(im), self._cv2.COLOR_RGB2BGR)


def _draw_tracking_ui(
    display,
    overlay: _PILTextOverlay,
    *,
    recording: bool,
    frames_written: int,
    record_elapsed: float,
    stale: bool,
    width: int,
    height: int,
) -> None:
    """底栏快捷键说明 + 录制状态 + 取流暂断提示（写入视频前调用，与预览一致）。"""
    h, w = display.shape[:2]
    if stale:
        st = "(上一帧 — 取流暂断)"
        sf = max(14, min(20, w // 64))
        tw, _ = overlay.text_size(st, sf)
        overlay.draw(
            display,
            st,
            (max(4, w - tw - 8), 6),
            sf,
            (255, 200, 0),
            stroke=1,
            stroke_bgr=(0, 0, 0),
        )

    status_f = max(18, min(26, h // 30))
    if recording:
        cv2 = overlay._cv2
        cv2.circle(display, (22, 22), 10, (0, 0, 255), -1)
        rec_line = f"录制可视化中 {record_elapsed:>5.1f}s  帧数:{frames_written}"
        overlay.draw(
            display,
            rec_line,
            (40, 4),
            status_f,
            (0, 0, 255),
            stroke=1,
            stroke_bgr=(40, 40, 40),
        )
    else:
        overlay.draw(
            display,
            "待机 — 按 r 录制当前画面（含跟踪标注）",
            (8, 4),
            status_f,
            (0, 220, 0),
            stroke=1,
            stroke_bgr=(20, 20, 20),
        )

    lines = [
        "[r] 开始录制  [s] 保存可视化  [空格] 取消本段  [q] 退出",
        "保存内容与窗口一致（检测框/分割/轨迹等）",
    ]
    font_px = max(16, min(22, w // 55))
    line_skip = font_px + 6
    line_y0 = min(height - 44, h - line_skip * 2 - 8)
    for i, text in enumerate(lines):
        overlay.draw(display, text, (8, line_y0 + i * line_skip), font_px, (240, 240, 240))


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_config = project_root / "fiveages_cameras" / "orbbec" / "orbbec_camera_usb_config.yaml"

    p = argparse.ArgumentParser(description="奥比中光相机 + YOLO 实时跟踪")
    p.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Orbbec YAML（默认: {default_config}）",
    )
    p.add_argument("--serial", type=str, default=None, help="覆盖配置文件中的 serial_number")
    p.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics 检测权重（本地路径或官方权重名，由 ultralytics 自动下载）",
    )
    p.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="跟踪器配置名或路径（默认: bytetrack.yaml）",
    )
    p.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    p.add_argument("--device", type=str, default=None, help="如 cuda:0 或 cpu；默认由 Ultralytics 自动选择")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "recordings",
        help="交互录制保存目录（默认：当前工作目录下的 recordings）",
    )
    p.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="VideoWriter 四字符编码，如 mp4v、XVID",
    )
    p.add_argument(
        "--font",
        type=Path,
        default=None,
        help="中文字体 .ttf/.ttc；默认同录制脚本自动查找",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="不打开窗口（无 r/s 录制；用于压测帧率）",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="处理帧数上限，0 表示直到按 q 退出",
    )
    p.add_argument(
        "--trails",
        action="store_true",
        help="在检测框中心绘制短时轨迹线（参考 material.txt）",
    )
    p.add_argument("--trail-len", type=int, default=30, help="轨迹保留帧数")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent.parent
    _setup_fiveages_cam_imports(project_root)

    import cv2
    import numpy as np
    from collections import defaultdict

    from orbbec.orbbec_camera_usb import OrbbecCameraUSB
    from ultralytics import YOLO

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        print(f"错误: 配置文件不存在: {config_path}", file=sys.stderr)
        return 1

    fourcc_str = args.fourcc
    if len(fourcc_str) != 4:
        print("错误: --fourcc 必须为 4 个字符", file=sys.stderr)
        return 1

    print(f"加载模型: {args.model} …", flush=True)
    model = YOLO(args.model)

    track_kw: dict = {
        "persist": True,
        "tracker": args.tracker,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
    }
    if args.device:
        track_kw["device"] = args.device

    camera = None
    window = "Orbbec + YOLO Tracking | r录制 s保存 空格取消 q退出"
    track_history: dict[int, list] = defaultdict(list) if args.trails else {}
    frame_i = 0
    t0 = time.perf_counter()
    infer_ms_ema: float | None = None

    writer = None
    temp_path: str | None = None
    recording = False
    record_start_mono: float | None = None
    frames_written = 0
    out_dir = (args.output_dir or Path.cwd()).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        serial_override = {"serial_number": args.serial} if args.serial else None
        print(f"打开相机: {config_path}", flush=True)
        camera = OrbbecCameraUSB(config=serial_override, config_path=str(config_path))

        width = int(camera.config.get("width", 640))
        height = int(camera.config.get("height", 480))
        fps = int(camera.config.get("fps", 30))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

        overlay = _PILTextOverlay(cv2, np, args.font)

        if args.no_show:
            print(
                f"实时跟踪 {width}x{height} | tracker={args.tracker} | 无窗口模式（不支持 r/s 录制）",
                flush=True,
            )
        else:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            try:
                cv2.resizeWindow(window, min(1280, width), min(720, height))
            except cv2.error:
                pass
            print(
                f"实时跟踪 {width}x{height} | tracker={args.tracker}\n"
                f"  保存目录: {out_dir}\n"
                "  [r] 开始录制可视化  [s] 保存  [空格] 取消本段  [q] 退出\n",
                flush=True,
            )

        last_display = None
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        boot_f = max(22, min(30, height // 26))
        overlay.draw_centered(
            placeholder,
            "正在拉取摄像头画面…",
            (width // 2, height // 2),
            boot_f,
            (255, 200, 0),
            stroke=1,
            stroke_bgr=(0, 0, 0),
        )

        while True:
            packet = camera.capture()
            frame_bgr = None
            fresh_inference = False

            if packet is not None:
                frame_bgr = _packet_to_bgr(packet, width, height, cv2)

            if frame_bgr is not None:
                t_inf0 = time.perf_counter()
                results = model.track(frame_bgr, **track_kw)
                t_inf = (time.perf_counter() - t_inf0) * 1000.0
                infer_ms_ema = t_inf if infer_ms_ema is None else 0.9 * infer_ms_ema + 0.1 * t_inf

                result = results[0]
                display = result.plot()

                if args.trails and result.boxes is not None and result.boxes.is_track:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id
                    if track_ids is not None:
                        track_ids_list = track_ids.int().cpu().tolist()
                        for box, tid in zip(boxes, track_ids_list):
                            x, y, w, h = box
                            tr = track_history[tid]
                            tr.append((float(x), float(y)))
                            if len(tr) > args.trail_len:
                                tr.pop(0)
                            pts = np.hstack(tr).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(display, [pts], isClosed=False, color=(180, 220, 255), thickness=2)

                if infer_ms_ema is not None:
                    cv2.putText(
                        display,
                        f"infer ~{infer_ms_ema:.1f} ms",
                        (8, height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                last_display = display.copy()
                fresh_inference = True
                stale = False
            else:
                if last_display is not None:
                    display = last_display.copy()
                    stale = True
                else:
                    display = placeholder.copy()
                    stale = False

            record_elapsed = time.monotonic() - record_start_mono if record_start_mono else 0.0
            _draw_tracking_ui(
                display,
                overlay,
                recording=recording,
                frames_written=frames_written,
                record_elapsed=record_elapsed,
                stale=stale,
                width=width,
                height=height,
            )

            if recording and fresh_inference and writer is not None and writer.isOpened():
                writer.write(display)
                frames_written += 1

            if not args.no_show:
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    if recording:
                        _discard_recording(writer, temp_path, cv2)
                        writer = None
                        temp_path = None
                        recording = False
                        frames_written = 0
                    break

                if key == ord("r"):
                    if recording:
                        pass
                    else:
                        fd, temp_path = tempfile.mkstemp(suffix=".mp4", dir=str(out_dir))
                        os.close(fd)
                        writer = cv2.VideoWriter(temp_path, fourcc, float(fps), (width, height))
                        if not writer.isOpened():
                            print("错误: VideoWriter 无法打开，请检查 --fourcc 或路径权限。", file=sys.stderr)
                            try:
                                os.unlink(temp_path)
                            except OSError:
                                pass
                            temp_path = None
                            writer = None
                        else:
                            recording = True
                            record_start_mono = time.monotonic()
                            frames_written = 0
                            print(">>> 正在录制可视化画面（按 s 保存，空格取消）", flush=True)

                elif key == ord("s"):
                    if not recording or writer is None or not temp_path:
                        pass
                    else:
                        writer.release()
                        writer = None
                        end_wall = time.time()
                        duration_sec = (
                            time.monotonic() - record_start_mono if record_start_mono is not None else 0.0
                        )
                        record_start_mono = None
                        recording = False

                        if frames_written == 0:
                            try:
                                os.unlink(temp_path)
                            except OSError:
                                pass
                            temp_path = None
                            frames_written = 0
                            print("未写入任何帧，已丢弃。", flush=True)
                        else:
                            final_name = _make_save_filename(end_wall, duration_sec)
                            final_path = out_dir / final_name
                            if final_path.is_file():
                                final_path.unlink()
                            os.replace(temp_path, str(final_path))
                            temp_path = None
                            frames_written = 0
                            print(f"已保存: {final_path}（约 {duration_sec:.2f} 秒）", flush=True)

                elif key == ord(" "):
                    if not recording:
                        pass
                    else:
                        _discard_recording(writer, temp_path, cv2)
                        writer = None
                        temp_path = None
                        recording = False
                        record_start_mono = None
                        frames_written = 0
                        print(">>> 已取消本段录制（未保存）", flush=True)

            if fresh_inference:
                frame_i += 1
            if args.max_frames > 0 and frame_i >= args.max_frames:
                break

        elapsed = time.perf_counter() - t0
        if frame_i > 0:
            print(f"共处理 {frame_i} 帧，用时 {elapsed:.2f}s，平均 {frame_i / elapsed:.1f} fps", flush=True)
        return 0

    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        if recording and writer is not None:
            _discard_recording(writer, temp_path, cv2)
        return 130
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if writer is not None:
            try:
                writer.release()
            except cv2.error:
                pass
        if temp_path and os.path.isfile(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
