from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

# 장바구니 테이블
class CartItem(Base):
    __tablename__ = "cart"

    id         = Column(Integer, primary_key=True, index=True)
    label      = Column(String)   # YOLO 클래스명 (candy, item_a ...)
    name       = Column(String)   # 한글 상품명
    price      = Column(Integer)  # 가격
    emoji      = Column(String)   # 이모지
    qty        = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())

# 구매 이력 테이블
class PurchaseLog(Base):
    __tablename__ = "purchase_log"

    id         = Column(Integer, primary_key=True, index=True)
    label      = Column(String)
    name       = Column(String)
    price      = Column(Integer)
    qty        = Column(Integer)
    total      = Column(Integer)
    created_at = Column(DateTime, default=func.now())
