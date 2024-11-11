デモバージョン配布

데모버전을 배포한다. 개발환경 : macOS 사용설명서는 다음과 같습니다.

가상환경 세팅 **** 가상환경의 경로는 꼴리는대로 설정해주십시오. ****
requirements.txt 에서 필요한 라이브러리를 설치합니다. copy >>> pip install -r requirements.txt

다 설치하세요
불안하다면 requirements 설치(mac os).txt 파일을 참조하십시오.
필요시 설치하세요 (macOS 에서 발생한 에러 입니다.)
추가적으로 에러 발생시 아래 라이브러리를 확인하세요.
pip install boost

pip install dlib

pip install python-multipart

pip install numpy==1.24.4

서버 설명서.txt 를 참조하여 FastAPI 작동 방법을 확인하세요.
터미널에서 가상환경에서 작동시키는법 (FastAPI 설치 되었을 시) ver.choi

cd /Users/DFD_capstone

source DFD_capstone/bin/activate

uvicorn main:app

작동시 아래 문구 출력 INFO: Started server process [42287]

INFO: Waiting for application startup.

INFO: Application startup complete.

INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

위 url 클릭 or 브라우저에 입력 합니다.

your_project/
├── main.py
├── modules/
│ ├── init.py
│ ├── model_loader.py
│ ├── video_processing.py
│ ├── session_manager.py
│ ├── utils.py
│ ├── network/
│ │ ├── init.py
│ │ └── models.py
│ └── dataset/
│ ├── init.py
│ └── transform.py
├── templates/
│ ├── index.html
│ ├── upload.html
│ ├── select_model.html
│ ├── loading.html
│ └── results.html
├── static/
│ └── (static files)
├── uploaded_videos/
├── requirements.txt
└── README.md