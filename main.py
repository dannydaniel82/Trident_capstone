import asyncio
import os
import shutil
import uuid
import cv2
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import subprocess
import numpy as np

from modules.model_loader import ModelLoader
from modules.video_processing import VideoProcessor
from modules.session_manager import SessionManager
import time

# 세션 매니저 생성
session_manager = SessionManager()

# FastAPI 앱 생성
app = FastAPI()
templates = Jinja2Templates(directory="templates")  # for HTML

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 업로드된 비디오 저장 디렉토리
video_directory = "uploaded_videos"
os.makedirs(video_directory, exist_ok=True)


# 1. 첫 번째 화면: 프로젝트 소개 및 '시작하기' 버튼
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# '시작하기' 버튼 클릭 처리
@app.post("/start")
async def start():
    return RedirectResponse(url="/upload", status_code=303)


# 2. 두 번째 화면: 비디오 업로드
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


# 비디오 업로드 처리
@app.post("/upload_video")
async def upload_video(request: Request, video: UploadFile = File(...)):
    # 세션 ID 생성
    session_id = session_manager.create_session()
    sessions = session_manager.sessions

    # 비디오 저장
    video_path = os.path.join(video_directory, f"{session_id}_{video.filename}")
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # 비디오 파일 크기 및 길이 제한 확인
    video_size = os.path.getsize(video_path) / (1024 * 1024)  # MB 단위
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / fps

    # 제한 조건 확인
    if video_duration > 3000 or video_size > 5000:
        cap.release()
        os.remove(video_path)
        raise HTTPException(status_code=400, detail="30초, 50MB 이하 동영상을 업로드 하세요")

    # 비디오 형식 확인 및 변환
    if not video.filename.endswith(('.mp4', '.mov')):
        converted_video_path = os.path.join(video_directory, f"{session_id}_converted.mp4")
        try:
            subprocess.run(["ffmpeg", "-i", video_path, "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
                            converted_video_path], check=True)
            os.remove(video_path)  # 원본 비디오 파일 삭제
            video_path = converted_video_path
        except subprocess.CalledProcessError:
            raise HTTPException(status_code=400, detail="비디오 변환 실패: mp4 또는 mov 형식의 비디오를 업로드해주세요.")

    # 비디오 썸네일 생성 (첫 번째 프레임)
    ret, frame = cap.read()
    if ret:
        thumbnail_path = os.path.join("static", f"{session_id}_thumbnail.jpg")
        cv2.imwrite(thumbnail_path, frame)
        session_manager.update_session(session_id, {'thumbnail': f"/{thumbnail_path}"})
    cap.release()

    # 세션 정보 저장
    session_manager.update_session(session_id, {'video_path': video_path})

    # 모델 선택 페이지로 이동
    return RedirectResponse(url=f"/select_model?session_id={session_id}", status_code=303)


# 3. 모델 선택 페이지
@app.get("/select_model", response_class=HTMLResponse)
async def select_model_page(request: Request, session_id: str):
    return templates.TemplateResponse("select_model.html", {"request": request, "session_id": session_id})


# 모델 및 프레임 설정 처리
@app.post("/select_model")
async def select_model(
        request: Request,
        session_id: str = Form(...),
        model_name: str = Form(...),
        frame_rate: str = Form(...)
):
    FRAME_RATE_MAPPING = {
        'low': 4,  # 5번째 프레임마다 frame extracting (Hard)
        'medium': 16,  # 15번째 프레임마다 frame extracting (Medium)
        'high': 32  # 30번째 프레임마다 frame extracting (Easy)
    }
    frame_rate_value = FRAME_RATE_MAPPING.get(frame_rate)
    # 모델 로딩
    model_loader = ModelLoader(model_name)
    model = model_loader.model

    # 세션 정보 업데이트
    session_manager.update_session(session_id, {
        'model_name': model_name,
        'frame_rate': frame_rate_value,
        'model': model
    })

    # 로딩 페이지로 이동하여 모델 처리 시작
    return RedirectResponse(url=f"/processing?session_id={session_id}", status_code=303)


# 4. 로딩 페이지 및 모델 처리
@app.get("/processing", response_class=HTMLResponse)
async def processing(request: Request, session_id: str):
    # 모델 처리를 백그라운드에서 수행
    asyncio.create_task(run_model(session_id))
    return templates.TemplateResponse("loading.html", {"request": request, "session_id": session_id})


# 결과 준비 여부 확인 엔드포인트
@app.get("/check_result")
async def check_result(session_id: str):
    sessions = session_manager.sessions
    if session_id in sessions:
        if sessions[session_id].get('result_ready'):
            return JSONResponse({"result_ready": True})
        elif sessions[session_id].get('error'):
            return JSONResponse({"result_ready": False, "error": sessions[session_id]['error']})
    return JSONResponse({"result_ready": False})


# 진행률 확인 엔드포인트
@app.get("/get_progress")
async def get_progress(session_id: str):
    sessions = session_manager.sessions
    if session_id in sessions:
        progress = sessions[session_id].get('progress', 0)
        return JSONResponse({"progress": progress})
    else:
        return JSONResponse({"progress": 0})


# 5. 결과 페이지
@app.get("/results", response_class=HTMLResponse)
async def results(request: Request, session_id: str):
    sessions = session_manager.sessions

    # 세션 존재 여부 확인
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # 에러 상태 확인
    if sessions[session_id].get('status') == 'error':
        time.sleep(10)
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error_message": sessions[session_id].get('error', "An unknown error occurred")
        })

    # 처리 완료 여부 확인
    if sessions[session_id].get('status') != 'completed':
        return RedirectResponse(url=f"/processing?session_id={session_id}")
    time.sleep(3)
    try:
        # 결과 데이터 가져오기
        thumbnail = sessions[session_id]['thumbnail']
        graph = sessions[session_id].get('graph')
        output_video = sessions[session_id]['output_video']
        frame_scores = sessions[session_id]['frame_scores']
        result = sessions[session_id]['result']
        label = sessions[session_id]['label']
        # calculate average
        try:
            average = f"{sessions[session_id]['average']:.2f}"
        except KeyError:
            if frame_scores:
                average = sum(frame_scores) / len(frame_scores)
                sessions[session_id]['average'] = average
            else:
                average = -1
        average = f"{float(average)*100:.2f}"
        return templates.TemplateResponse("results.html", {
            "request": request,
            "session_id": session_id,
            "result": result,
            "thumbnail": thumbnail,
            "graph": graph,
            "output_video": output_video,
            "average": average,
            "label": label
        })
    except KeyError as e:
        # 필요한 결과 데이터가 없는 경우
        missing_key = str(e).strip("'")
        error_message = f"Missing required result data: {missing_key}"
        sessions[session_id]['error'] = error_message
        sessions[session_id]['status'] = 'error'
        time.sleep(5)
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error_message": error_message
        })

# CSV 생성 및 다운로드 엔드포인트
@app.get("/download_csv")
async def download_csv(session_id: str):
    sessions = session_manager.sessions
    frame_scores = sessions[session_id]['frame_scores']
    avg_score = sum(frame_scores) / len(frame_scores)
    result = sessions[session_id]['result']
    csv_path = f"static/{session_id}_scores.csv"
    with open(csv_path, 'w') as f:
        f.write("Frame Number,Score\n")
        for idx, score in enumerate(frame_scores):
            f.write(f"{idx},{score:.4f}\n")
        f.write(f"\nAverage Score,{avg_score:.4f}\n")
        f.write(f"Result,{result}\n")

    return FileResponse(csv_path, media_type='text/csv', filename='Deepfake_Detection_Result.csv')


# 종료하기 엔드포인트
@app.post("/reset")
async def reset(session_id: str = Form(...)):
    # 세션 데이터 삭제
    session_manager.delete_session(session_id)
    return RedirectResponse(url="/", status_code=303)


# 비디오 제공 엔드포인트
@app.get("/video/{session_id}")
async def get_video(session_id: str):
    final_output_video_path = os.path.join("static", f"{session_id}_final_output.mp4")
    if os.path.exists(final_output_video_path):
        return FileResponse(final_output_video_path, media_type='video/mp4')
    else:
        raise HTTPException(status_code=404, detail="Video not found")


def process_video(session_id):
    sessions = session_manager.sessions
    try:
        # 세션 상태를 processing으로 설정
        sessions[session_id]['status'] = 'processing'

        # 비디오 처리
        processor = VideoProcessor(session_id, sessions)
        processor.process()

        # 처리 완료 후 상태 업데이트
        sessions[session_id]['status'] = 'completed'

    except Exception as e:
        # 상세한 에러 처리
        import traceback
        error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        sessions[session_id]['error'] = error_message
        sessions[session_id]['status'] = 'error'
        sessions[session_id]['result_ready'] = False
        print(error_message)
        # 에러 발생 시 진행률을 0으로 리셋
        sessions[session_id]['progress'] = 0


async def run_model(session_id):
    try:
        await asyncio.to_thread(process_video, session_id)
    except Exception as e:
        # 비동기 처리 중 발생하는 에러 처리
        sessions = session_manager.sessions
        sessions[session_id]['error'] = f"Processing error: {str(e)}"
        sessions[session_id]['status'] = 'error'
        sessions[session_id]['result_ready'] = False
        sessions[session_id]['progress'] = 0
