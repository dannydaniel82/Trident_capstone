# video_processing.py

import cv2
import dlib
import torch
import subprocess
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import get_boundingbox, predict_with_model

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

    def extract_audio(self):
        # 오디오 추출
        audio_path = os.path.join("static", f"{self.session_id}_audio.aac")
        command = [
            'ffmpeg', '-y', '-i', self.video_path, '-vn',
            '-acodec', 'copy', audio_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            error_message = f"Audio extraction failed: {result.stderr.decode()}"
            print(error_message)
            raise Exception(error_message)
        else:
            print("Audio extraction successful.")
        self.audio_path = audio_path

    def process_video(self):
        # Initialize video reader
        reader = cv2.VideoCapture(self.video_path)
        num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = reader.get(cv2.CAP_PROP_FPS)
        frame_scores = []
        frame_num = 0

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
            if not ret:
                break
            frame_num += 1
            pbar.update(1)  # Update the progress bar once per frame

            # 진행률 계산 및 세션에 저장
            progress = int((frame_num / num_frames) * 100)
            self.sessions[self.session_id]['progress'] = progress

            # Process only every nth frame based on frame_rate
            if frame_num % self.frame_rate == 0:
                # Convert image to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector(gray, 1)

                if len(faces):
                    # Use the first detected face
                    face = faces[0]
                    x1, y1, size = get_boundingbox(face, image.shape[1], image.shape[0])
                    cropped_face = image[y1:y1 + size, x1:x1 + size]

                    # Predict with model
                    prediction, output = predict_with_model(cropped_face, self.model, cuda=self.device.type == 'cuda')
                    score = output[0][1].item()  # Assuming output is [batch_size, 2] with [real_score, fake_score]
                    frame_scores.append(score)

                    # Determine label and color based on prediction
                    label = 'Fake' if prediction == 1 else 'Real'
                    color = (0, 0, 255) if prediction == 1 else (0, 255, 0)

                    # Format output_list
                    output_list = ['{0:.2f}'.format(float(x)) for x in output.detach().cpu().numpy()[0]]

                    # Update last known annotations
                    last_x, last_y, last_w, last_h = x1, y1, size, size
                    last_label = label
                    last_color = color
                    last_output_list = output_list
                else:
                    pass
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

        self.output_video_path = output_video_path
        self.frame_scores = frame_scores
        self.fps = fps

    def merge_audio_video(self):
        # 비디오 처리 완료 후 오디오와 비디오 결합
        final_output_video_path = os.path.join("static", f"{self.session_id}_final_output.mp4")
        command = [
            'ffmpeg', '-y', '-i', self.output_video_path, '-i', self.audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', '-shortest', final_output_video_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            error_message = f"Video and audio merging failed: {result.stderr.decode()}"
            print(error_message)
            raise Exception(error_message)
        else:
            print("Video and audio merging successful.")
        self.final_output_video_path = final_output_video_path

        # 임시 파일 삭제
        os.remove(self.output_video_path)
        os.remove(self.audio_path)

    def save_results(self):
        # Calculate average score
        if self.frame_scores:
            avg_score = sum(self.frame_scores) / len(self.frame_scores)
        else:
            avg_score = 0.5  # Assign a neutral average score if no frames were processed
        label = 'Real' if avg_score < 0.5 else 'Fake'

        # 세션에 결과 저장
        self.sessions[self.session_id]['progress'] = 100
        self.sessions[self.session_id]['frame_scores'] = self.frame_scores
        self.sessions[self.session_id]['result'] = f'Average Score: {avg_score:.2f} - {label}'
        self.sessions[self.session_id]['output_video'] = f"/{self.final_output_video_path}"
        self.sessions[self.session_id]['result_ready'] = True

    def generate_graph(self):
        # 그래프 생성 및 저장
        graph_path = f"static/{self.session_id}_graph.png"
        plt.figure(figsize=(10, 4))
        plt.plot(self.frame_scores, marker='o')
        plt.title('Frame-wise Score')
        plt.xlabel('Frame Index')
        plt.ylabel('Fake Probability')
        plt.savefig(graph_path)
        plt.close()
        self.sessions[self.session_id]['graph'] = f"/{graph_path}"
    def process(self):
        self.extract_audio()
        self.process_video()
        self.merge_audio_video()
        self.save_results()
        self.generate_graph()

    def generate_graph(self):
        import matplotlib
        matplotlib.use('Agg')  # 백엔드를 'Agg'로 설정하여 GUI 사용 방지
        import matplotlib.pyplot as plt

        # 그래프 생성 및 저장
        graph_path = f"static/{self.session_id}_graph.png"
        plt.figure(figsize=(10, 4))
        plt.plot(self.frame_scores, marker='o')
        plt.title('Frame-wise Score')
        plt.xlabel('Frame Index')
        plt.ylabel('Fake Probability')
        plt.savefig(graph_path)
        plt.close()
        self.sessions[self.session_id]['graph'] = f"/{graph_path}"



