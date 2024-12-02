# Dankook University
# 단국대학교 소프트웨어학과 캡스톤디자인
# 딥페이크 탐지 솔루션

# Model :: Xception
reference1 :: https://github.com/HongguLiu/Deepfake-Detection
reference2 :; https://github.com/ondyari/FaceForensics

## Python, FastAPI, HTML, CSS


개발환경 : macOS 인 점에 유의하여, 각종 라이브러리에 오류가 발생할 시 requirements.txt 를 확인해주세요

## 가상환경 세팅
**** 가상환경의 경로는 사용자 커스텀으로 설정해주십시오. ****

requirements.txt 에서 필요한 라이브러리를 설치합니다.

copy >>> pip install -r requirements.txt

### 예외처리를 위한 requirements(macos).txt 파일을 참고하세요

pip install boost
pip install dlib
pip install python-multipart
pip install numpy==1.24.4

# FastAPI 작동 방법
서버 설명서.txt 를 참조하여 FastAPI 작동 방법을 확인하세요.
터미널에서 가상환경에서 작동시키는법 (FastAPI 설치 되었을 시)

cd /Users/DFD_capstone

source DFD_capstone/bin/activate

uvicorn main:app

작동시 아래 문구 출력 INFO: Started server process [42287]

INFO: Waiting for application startup.

INFO: Application startup complete.

INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

위 url 클릭 or 브라우저에 입력 합니다.
