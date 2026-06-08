from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data0608_2_aug/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=15,

    project="0608output",
    name="fall_detect_v1",

    exist_ok=True
)