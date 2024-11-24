# video_processing.py

import cv2
import dlib
import torch
import subprocess
import os
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from .utils import get_boundingbox, predict_with_model

from dataset.transform import xception_default_data_transforms
from PIL import Image as pil_image
import time
class VideoProcessor:
    def __init__(self, session_id, sessions):
        self.session_id = session_id
        self.sessions = sessions
        self.video_path = sessions[session_id]['video_path']
        self.frame_rate = sessions[session_id]['frame_rate']
        self.model = sessions[session_id]['model']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detector = dlib.get_frontal_face_detector()
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.thickness = 2
        self.font_scale = 1
        self.audio_path = None

    # 비디오의 음성 스트림이 있는지 확인, 없다면 추출 및 병합과정 생략
    def has_audio_stream(self):
        command = ['ffprobe', '-i', self.video_path, '-show_streams', '-select_streams', 'a', '-loglevel', 'error']
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return bool(result.stdout.strip())  # 오디오 스트림이 있으면 True 반환


    def extract_audio(self):
        print("시작: 오디오 추출 중...")
        self.audio_path = os.path.join("static", f"{self.session_id}_audio.aac")
        command = ['ffmpeg', '-y', '-i', self.video_path, '-vn', '-acodec', 'copy', self.audio_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("오류: 오디오 추출 실패")
        print("완료: 오디오 추출 성공")

    def merge_audio_video(self):
        if not self.audio_path: # 오디오 없는 파일일시, merge 건너뛰기
            self.final_output_path = self.output_video_path
            return

        print("시작: 오디오와 비디오 병합 중...")
        final_output_video_path = os.path.join("static", f"{self.session_id}_final_output.mp4")
        command = [
            'ffmpeg', '-y', '-i', self.output_video_path, '-i', self.audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', '-shortest',
            final_output_video_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0: # 병합실패시 비디오만 사용
            self.final_output_video_path = self.output_video_path
        else:
            print("완료: 오디오/비디오 병합 성공")
            self.final_output_video_path = final_output_video_path


    def preprocess_image(image, cuda=True):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocess = xception_default_data_transforms['test']
        preprocessed_image = preprocess(pil_image.fromarray(image))
        preprocessed_image = preprocessed_image.unsqueeze(0)
        if cuda:
            preprocessed_image = preprocessed_image.cuda()
        return preprocessed_image
    def process_video(self):
        try:
            # 상태를 processing 으로 변경
            if self.sessions[self.session_id].get('status') == 'completed':
                self.output_video_path = os.path.join("static", f"{self.session_id}_output.mp4")
                print('이미 처리된 세션입니다.')
                return
            self.sessions[self.session_id]['status'] = 'processing'

            # Initialize video reader
            reader = cv2.VideoCapture(self.video_path)
            num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = reader.get(cv2.CAP_PROP_FPS)
            frame_scores = []
            frame_num = 0
            smoothed_score_result = []
            # Initialize video writer
            output_video_path = os.path.join("static", f"{self.session_id}_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
            width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            # Initialize last known annotations
            last_x, last_y, last_w, last_h = None, None, None, None
            last_label = None
            last_color = (0, 255, 0)  # Default color (e.g., Green for 'Real')
            last_output_list = None

            # Process video frames
            pbar = tqdm(total=num_frames)
            while reader.isOpened():
                ret, image = reader.read()
                #if not ret:
                if image is None:
                    break
                frame_num += 1
                pbar.update(1)  # Update the progress bar once per frame

                # 진행률 계산 및 세션에 저장
                progress = int((frame_num / num_frames) * 100)
                self.sessions[self.session_id]['progress'] = progress

                # 1122 수정작업
                smoothing_window = 5
                frame_score_history = []
                stability_threshold = 0.2 # 점수 변동성을 허용하는 범위

                ############
                # Process score every nth frame based on frame_rate
                if frame_num % self.frame_rate == 0:
                    # Convert image to grayscale for face detection
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector(gray, 1)
                    #if len(faces):
                    if len(faces) > 0:
                        # Use the first detected face
                        # error concept
                        face = faces[0]
                        x1, y1, size = get_boundingbox(face, image.shape[1], image.shape[0])
                        cropped_face = image[y1:y1 + size, x1:x1 + size]

                        # Predict with model
                        prediction, output = predict_with_model(image, self.model,
                                                                cuda=self.device.type == 'cuda')
                        score = output[0][1].item()  # Assuming output is [batch_size, 2] with [real_score, fake_score]
                        print(score)

                        ### 점수 안정화 로직 11.22###
                        frame_score_history.append(score)
                        if len(frame_score_history) > smoothing_window:
                            frame_score_history.pop(0)
                        ### 이동평균 계산 ###
                        smoothed_score = sum(frame_score_history) / len(frame_score_history)
                        score_variance = max(frame_score_history) - min(frame_score_history)
                        smoothed_score_result.append(smoothed_score)
                        frame_scores.append(score)
                        # 안정성 기반 레이블링
                        if score_variance <= stability_threshold:
                            label = 'Fake' if smoothed_score > 0.5 else 'Real'
                            color = (0, 0, 255) if label == 'Fake' else (0, 255, 0)
                        else:
                            label = 'Uncertain'
                            color = (255, 255, 0)  # Yellow for uncertainty
                        # Format output_list
                        output_list = ['{0:.2f}'.format(float(x)) for x in output.detach().cpu().numpy()[0]]

                        # Update last known annotations
                        last_x, last_y, last_w, last_h = x1, y1, size, size
                        last_label = label
                        last_color = color
                        last_output_list = output_list

                # Apply last known annotations to the current frame
                if last_x is not None and last_y is not None and last_w is not None and last_h is not None:
                    # Annotate with last known output_list and label
                    if last_output_list:
                        cv2.putText(image, str(last_output_list) + ' => ' + last_label, (last_x, last_y + last_h + 30),
                                    self.font_face, self.font_scale, last_color, self.thickness, 2)
                    # Draw the bounding box
                    cv2.rectangle(image, (last_x, last_y), (last_x + last_w, last_y + last_h), last_color, 2)
                else:
                    # Annotate  "Non-Detected"
                    cv2.putText(image, "Non-Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 255, 255), 2)

                # Write the annotated frame to the output video
                writer.write(image)

            pbar.close()
            reader.release()
            writer.release()
            print(f"Total number of frames: {num_frames}")
            print(smoothed_score_result)
            average = sum(frame_scores) / len(frame_scores)
            label = 'Real' if average < 0.5 else 'Fake'
            self.average = average
            self.label = label
            self.frame_scores = frame_scores
            # 세션 상태를 completed 로 변경
            self.sessions[self.session_id]['output_video'] = f"/{output_video_path}"
            self.output_video_path = output_video_path
            self.fps = fps
            self.sessions[self.session_id]['status'] = 'completed'

        except Exception as e:
            # 세션 상태를 error 로 설정하고 예외 메시지를 출력
            self.sessions[self.session_id]['status'] = 'error'
            print(f"비디오 처리 중 에러 발생: {str(e)}")
            raise


    def generate_graph(self):
        print("시작: 그래프 생성 중...")
        matplotlib.use('Agg')

        # 심해 테마 색상 설정
        plt.style.use('dark_background')
        colors = {
            'background': '#0A1931',  # 깊은 바다색
            'grid': '#466093',
            'line': '#00F5FF',  # 형광 블루
            'marker': '#7DF9FF'  # 하늘빛 블루
        }

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(colors['background'])
        ax.set_facecolor(colors['background'])

        # 그리드 스타일 설정
        ax.grid(color=colors['grid'], linestyle='--', alpha=0.3)

        # 플롯 생성
        x = range(len(self.frame_scores))
        ax.plot(x, self.frame_scores,
                color=colors['line'],
                linewidth=2,
                alpha=0.8,
                label='Deepfake Score')

        # 마커 추가
        ax.scatter(x, self.frame_scores,
                   color=colors['marker'],
                   s=30,
                   alpha=0.6)

        # 스타일링
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(colors['grid'])
        ax.spines['left'].set_color(colors['grid'])

        # 라벨 설정
        ax.set_title('Trident Deepfake Detection Score',
                     color='white',
                     pad=20,
                     fontsize=12)
        ax.set_xlabel('Frame Index', color='white', labelpad=10)
        ax.set_ylabel('Fake Probability', color='white', labelpad=10)

        # 눈금 색상 설정
        ax.tick_params(colors='white')
        plt.tight_layout()
        graph_path = f"static/{self.session_id}_graph.png"
        plt.savefig(graph_path,
                    facecolor=colors['background'],
                    edgecolor='none',
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

        print(f"완료: 그래프 생성 완료 ({graph_path})")
        self.sessions[self.session_id]['graph'] = f"/{graph_path}"

    def save_results(self):
        print("시작: 결과 저장 중...")
        # 모든 처리가 완료되었는지 확인
        if not hasattr(self, 'frame_scores') or not hasattr(self, 'final_output_video_path'):
            print("경고: 필요한 처리가 완료되지 않았습니다.")
            return False
        # 모든 결과가 준비된 후에만 result_ready를 True로 설정
        self.sessions[self.session_id].update({
            'progress': 100,
            'frame_scores': self.frame_scores,
            'output_video': f"/{self.final_output_video_path}",
            'average': self.average,
            'label': self.label,
            'result': f'Average Score: {self.average:.2f} - {self.label}',
            'result_ready': True  # 마지막에 설정
        })

        print("완료: 결과 저장 완료")
        return True


    def process(self):
        print(f"\n{'=' * 50}\n작업 시작: 세션 ID {self.session_id}\n{'=' * 50}")
        has_audio = self.has_audio_stream()
        # audio processing
        if has_audio:
            audio_extracted = self.extract_audio()
        else:
            print("no audio stream detected")
            audio_extracted = False
        # video processing
        self.process_video()
        # audio merging
        if has_audio and audio_extracted:
            self.merge_audio_video()
        else:
            self.final_output_video_path = self.output_video_path
        # generate graph
        self.generate_graph()
        time.sleep(1)
        self.save_results()
        print(f"\n{'=' * 50}\n모든 작업 완료: 세션 ID {self.session_id}\n{'=' * 50}")

