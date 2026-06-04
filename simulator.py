import asyncio
import json
import math

async def simulate_objects(manager, process_tracking, db_get_cart):
    """
    테스트용 가상 객체 시뮬레이션
    - item_a 위→아래 2번 (구매)
    - item_a 아래→위 1번 (제거)
    """
    frame_h = 480
    frame_w = 640

    # 시나리오 정의
    scenarios = [
        {"id": 1, "label": "item_a", "direction": "down"},  # A→B→C 구매
        {"id": 2, "label": "item_a", "direction": "down"},  # A→B→C 구매
        {"id": 3, "label": "item_a", "direction": "up"},    # C→B→A 제거
    ]

    for s in scenarios:
        track_id = s["id"]
        label    = s["label"]

        if s["direction"] == "down":
            # 위→아래 (A→B→C)
            y_positions = range(0, frame_h, 10)
        else:
            # 아래→위 (C→B→A)
            y_positions = range(frame_h, 0, -10)

        for y in y_positions:
            center_y = y
            if center_y < frame_h / 3:
                zone = "A"
            elif center_y < frame_h * 2 / 3:
                zone = "B"
            else:
                zone = "C"

            # 감지 결과 전송
            await manager.broadcast({
                "type": "detections",
                "detections": [{
                    "track_id": track_id,
                    "label":    label,
                    "conf":     0.95,
                    "zone":     zone,
                    "bbox":     [200, y, 400, y+100]
                }],
                "frame_w": frame_w,
                "frame_h": frame_h
            })

            # 트래킹 처리
            result = process_tracking(track_id, zone, label)
            if result:
                cart  = db_get_cart()
                total = sum(i["price"]*i["qty"] for i in cart)
                await manager.broadcast({
                    "type":    "item_added" if result["action"]=="add" else "item_removed",
                    "product": result["product"],
                    "cart":    cart,
                    "total":   total,
                    "log":     f"{result['product']['name']} {'추가(A→B→C)' if result['action']=='add' else '제거(C→B→A)'}"
                })

            await asyncio.sleep(0.1)  # 이동 속도

        await asyncio.sleep(1)  # 다음 객체 전 대기
        print(f"✅ ID:{track_id} {s['direction']} 완료")