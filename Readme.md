# 1. 가상환경 생성
python3 -m venv venv

# 2. 활성화
source venv/bin/activate

pip install -r requirements.txt






uvicorn main:app --reload