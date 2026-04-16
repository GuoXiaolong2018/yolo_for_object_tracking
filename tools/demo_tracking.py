from ultralytics import YOLO

# Load an official or custom model
# model = YOLO("yolo26n.pt")  # Load an official Detect model
# model = YOLO("yolo26n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo26n-pose.pt")  # Load an official Pose model
model = YOLO("yolo11n.pt")  # 或改为你的本地权重路径

# Perform tracking with the model
# results = model.track("sample.mp4", show=True)  # Tracking with default tracker
results = model.track("sample.mp4", show=True, tracker="./bytetrack.yaml")  # with ByteTrack；请将 sample.mp4 换为你的视频