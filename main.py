import cv2
import json
import asyncio
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session

from database import engine, get_db, SessionLocal
from models import Base, CartItem, PurchaseLog

Base.metadata.create_all(bind=engine)

# ── 상품 정보 ──
PRODUCT_DB = {
    "item_a":    {"name": "상품A",    "price": 1000, "emoji": "📦"},
    "candy":     {"name": "사탕",     "price": 500,  "emoji": "🍬"},
    "chocolate": {"name": "초콜릿",   "price": 1200, "emoji": "🍫"},
    "bottle":    {"name": "음료수",   "price": 1500, "emoji": "🍶"},
    "cup":       {"name": "컵",       "price": 800,  "emoji": "☕"},
    "apple":     {"name": "사과",     "price": 2000, "emoji": "🍎"},
    "banana":    {"name": "바나나",   "price": 1300, "emoji": "🍌"},
    "free_time": {"name": "자유시간", "price": 1500, "emoji": "🍫"},
}

def get_product(label: str):
    return PRODUCT_DB.get(label.lower(), {"name": label, "price": 0, "emoji": "📦"})

# ── YOLO 모델 ──
model = None

def load_model():
    global model
    model_path = Path("best.pt")
    if model_path.exists():
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print("✅ YOLO 모델 로드 완료")
    else:
        print("⚠️  best.pt 없음")

# ── 스트리밍용 프레임 ──
latest_frame = None
frame_lock   = threading.Lock()

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ── 구역 판별 ──
def get_zone(center_y: float, frame_h: int) -> str:
    if center_y < frame_h / 3:       return "A"
    elif center_y < frame_h * 2 / 3: return "B"
    return "C"

# ── 트래킹 상태 ──
track_history: dict = {}
purchased:     set  = set()

# ── DB 함수 ──
def db_add_item(label: str) -> dict:
    product = get_product(label)
    db = SessionLocal()
    try:
        existing = db.query(CartItem).filter(CartItem.label == label).first()
        if existing:
            existing.qty += 1
            db.commit()
        else:
            db.add(CartItem(label=label, name=product["name"],
                            price=product["price"], emoji=product["emoji"], qty=1))
            db.commit()
        return product
    finally:
        db.close()

def db_get_cart() -> list:
    db = SessionLocal()
    try:
        return [{"id": i.id, "label": i.label, "name": i.name,
                 "price": i.price, "emoji": i.emoji, "qty": i.qty}
                for i in db.query(CartItem).all()]
    finally:
        db.close()

def db_remove_item(label: str):
    db = SessionLocal()
    try:
        item = db.query(CartItem).filter(CartItem.label == label).first()
        if item:
            if item.qty > 1: item.qty -= 1
            else: db.delete(item)
            db.commit()
    finally:
        db.close()

def db_clear_cart():
    db = SessionLocal()
    try:
        db.query(CartItem).delete()
        db.commit()
    finally:
        db.close()

def db_checkout():
    db = SessionLocal()
    try:
        for item in db.query(CartItem).all():
            db.add(PurchaseLog(label=item.label, name=item.name,
                               price=item.price, qty=item.qty,
                               total=item.price * item.qty))
        db.query(CartItem).delete()
        db.commit()
    finally:
        db.close()

# ── 트래킹 처리 ──
def process_tracking(track_id, zone: str, label: str) -> dict | None:
    if track_id not in track_history:
        track_history[track_id] = []
    hist = track_history[track_id]
    if not hist or hist[-1] != zone:
        hist.append(zone)

    ai = next((i for i, z in enumerate(hist) if z == "A"), -1)
    bi = next((i for i in range(len(hist)-1, -1, -1) if hist[i] == "B"), -1)
    ci = next((i for i in range(len(hist)-1, -1, -1) if hist[i] == "C"), -1)
    if ai != -1 and bi > ai and ci > bi:
        track_history[track_id] = []
        return {"action": "add", "product": db_add_item(label)}

    ci2 = next((i for i, z in enumerate(hist) if z == "C"), -1)
    bi2 = next((i for i in range(len(hist)-1, -1, -1) if hist[i] == "B"), -1)
    ai2 = next((i for i in range(len(hist)-1, -1, -1) if hist[i] == "A"), -1)
    if ci2 != -1 and bi2 > ci2 and ai2 > bi2:
        db_remove_item(label)
        track_history[track_id] = []
        return {"action": "remove", "product": get_product(label)}

    return None

# ── WebSocket 관리 ──
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data, ensure_ascii=False)
        for ws in self.active.copy():
            try:
                await ws.send_text(msg)
            except:
                self.active.remove(ws)

manager = ConnectionManager()

# ── YOLO 감지 루프 ──
async def yolo_detect_loop():
    global latest_frame
    if model is None:
        print("⚠️  YOLO 모델 없음")
        return

    VIDEO_PATH = "data/freetimetest.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH if Path(VIDEO_PATH).exists() else 0)
    print(f"🎬 영상 로드: {VIDEO_PATH}" if Path(VIDEO_PATH).exists() else "📷 카메라 시작")

    ZONE_COLORS = {"A": (147,139,250), "B": (56,189,248), "C": (52,211,153)}

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            await asyncio.sleep(0.1)
            continue

        frame_h, frame_w = frame.shape[:2]

        # 구역선
        cv2.line(frame, (0, frame_h//3),   (frame_w, frame_h//3),   (255,255,255), 1)
        cv2.line(frame, (0, frame_h*2//3), (frame_w, frame_h*2//3), (255,255,255), 1)
        cv2.putText(frame, "ZONE A", (10, 22),               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (147,139,250), 2)
        cv2.putText(frame, "ZONE B", (10, frame_h//3+22),   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56,189,248),  2)
        cv2.putText(frame, "ZONE C", (10, frame_h*2//3+22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (52,211,153),  2)

        results = model.track(frame, persist=True, verbose=False)
        detections = []

        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                track_id = int(box.id[0])
                label    = model.names[int(box.cls[0])]
                conf     = float(box.conf[0])
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                zone  = get_zone((y1+y2)/2, frame_h)
                color = ZONE_COLORS.get(zone, (255,255,255))

                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                cv2.putText(frame, f"{label} {int(conf*100)}% [{zone}]",
                            (int(x1), max(int(y1)-8,12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detections.append({"track_id": track_id, "label": label,
                                   "conf": round(conf,2), "zone": zone,
                                   "bbox": [x1,y1,x2,y2]})

                result = process_tracking(track_id, zone, label)
                if result:
                    cart  = db_get_cart()
                    total = sum(i["price"]*i["qty"] for i in cart)
                    await manager.broadcast({
                        "type":    "item_added" if result["action"]=="add" else "item_removed",
                        "product": result["product"],
                        "cart":    cart,
                        "total":   total,
                        "log":     f"{result['product']['name']} {'추가 (A→B→C)' if result['action']=='add' else '제거 (C→B→A)'}"
                    })

        with frame_lock:
            latest_frame = frame.copy()

        if detections and manager.active:
            await manager.broadcast({"type": "detections", "detections": detections,
                                     "frame_w": frame_w, "frame_h": frame_h})

        await asyncio.sleep(0.03)

    cap.release()

# ── 앱 시작 ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    asyncio.create_task(yolo_detect_loop())
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── 라우터 ──
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace;boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    cart = db_get_cart()
    await ws.send_text(json.dumps({"type": "cart_init", "cart": cart,
                                   "total": sum(i["price"]*i["qty"] for i in cart)},
                                  ensure_ascii=False))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg["type"] == "remove_item":
                db_remove_item(msg["label"])
                cart = db_get_cart()
                await manager.broadcast({"type": "cart_update", "cart": cart,
                                         "total": sum(i["price"]*i["qty"] for i in cart)})
            elif msg["type"] == "clear_cart":
                db_clear_cart()
                track_history.clear()
                await manager.broadcast({"type": "cart_update", "cart": [], "total": 0})
            elif msg["type"] == "checkout":
                cart  = db_get_cart()
                total = sum(i["price"]*i["qty"] for i in cart)
                db_checkout()
                track_history.clear()
                await manager.broadcast({"type": "checkout_done", "total": total, "cart": []})
    except WebSocketDisconnect:
        manager.disconnect(ws)

@app.get("/cart")
def get_cart(db: Session = Depends(get_db)):
    return db.query(CartItem).all()

@app.get("/")
def index():
    return HTMLResponse(open("static/index.html").read())
