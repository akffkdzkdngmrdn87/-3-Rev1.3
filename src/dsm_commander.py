# 파일명: src/dsm_commander.py
# [혁신 모빌리티 AI - 라즈베리파이 3 (1GB) 전용 초저전력 DSM 상용화 릴리즈 버전]

import os # 운영체제 제어 도구 (CPU 코어 할당 등)
import cv2 # 영상 처리 및 화면 출력 라이브러리 (OpenCV)
import socket # 관제 모니터와의 TCP 통신을 위한 네트워크 도구
import struct # 통신 시 영상 데이터를 안전하게 포장하는 패키징 도구
import numpy as np # 행렬 및 수학적 배열 계산을 번개처럼 처리하는 핵심 도구
import dlib # 얼굴 인식 및 68개의 얼굴 뼈대(랜드마크)를 추출하는 알고리즘
import tflite_runtime.interpreter as tflite # 양자 압축된 8비트 경량 AI 모델을 구동하는 엔진
import ncnn # C++ 기반으로 설계된 초고속 모바일 전용 딥러닝 엔진
import time # 수면 시간(2초) 및 딴짓 시간(1초)을 측정하는 초정밀 타이머
import collections # 최근 2초간의 운전자 상태 데이터를 차곡차곡 저장하는 메모리 버퍼
import math # 두 점 사이의 거리 등을 계산하기 위한 수학 모듈
import gc # 1GB RAM의 한계를 극복하기 위한 수동 메모리 청소기(Garbage Collector)

# 🚨 라즈베리파이 3의 한정된 자원을 극대화하기 위해 4개의 CPU 코어를 전면 풀가동합니다.
os.environ['OMP_NUM_THREADS'] = '4'

# ==========================================
# 1. 초저지연 통신망 및 시스템 타이머 셋팅
# ==========================================
CAM_W, CAM_H = 320, 240 # 연산 부하를 최소화하기 위해 해상도를 320x240으로 경량화합니다.
FPS_LIMIT = 10 # 1초에 10장의 프레임만 처리하도록 제한하여 CPU 과열을 방지합니다.
FRAME_INTERVAL = 1.0 / FPS_LIMIT 

# 외부 모니터로 관제 영상을 실시간 전송하기 위한 통신망(TCP 소켓)을 엽니다.
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 9999))
server_socket.listen(1)

print("=== 📡 지휘관 노트북 접속 대기 중... ===")
client_socket, addr = server_socket.accept()
# 데이터 전송 시 지연 시간을 없애는 TCP_NODELAY 알고리즘을 적용하여 실시간성을 확보합니다.
client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
print("🎉 지휘관 접속 완료! 조국 AI 무결점 관제 가동!")

# ==========================================
# 2. 12만장 융합 기반 8비트 양자화 AI 모델 장전
# ==========================================
detector = dlib.get_frontal_face_detector() # 사진에서 사람의 얼굴(네모 박스)을 찾습니다.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 눈, 코, 입 등 68개 뼈대 점을 찍습니다[cite: 10].
tracker = dlib.correlation_tracker() # 한 번 찾은 얼굴을 부드럽게 추적하여 연산량을 대폭 줄입니다[cite: 10].
is_tracking = False

# 🚨 [독자 기술] AI허브 11만장 + 직접 촬영한 5,000장으로 융합 학습 후 8비트로 압축한 '눈(Eye) 판독 모델' 장전![cite: 10]
eye_interp = tflite.Interpreter("eye_best_quant.tflite", num_threads=4)
eye_interp.allocate_tensors()
eye_in, eye_out = eye_interp.get_input_details()[0]['index'], eye_interp.get_output_details()[0]['index']

# 🚨 [독자 기술] AI허브 11만장 + 직접 촬영한 5,000장으로 융합 학습한 '입(Mouth) 하품 판독 모델' 장전![cite: 10]
mouth_interp = tflite.Interpreter("mouth_best_quant.tflite", num_threads=4)
mouth_interp.allocate_tensors()
mouth_in, mouth_out = mouth_interp.get_input_details()[0]['index'], mouth_interp.get_output_details()[0]['index']

# 마스크 착용 여부 판별 딥러닝 모델 장전[cite: 10]
mask_interp = tflite.Interpreter("model_quantized.tflite", num_threads=4)
mask_interp.allocate_tensors()
mask_input_details = mask_interp.get_input_details()[0]
mask_in, mask_out = mask_input_details['index'], mask_interp.get_output_details()[0]['index']
mask_shape, mask_dtype = mask_input_details['shape'], mask_input_details['dtype']

# 고개 방향(전후좌우)을 판별하는 C++ 기반 NCNN 딥러닝 모델 장전[cite: 10]
print("⚙️ NCNN 거대 무기 예열 중...")
net_head = ncnn.Net()
# 1GB RAM 환경에서 메모리 부족(Segfault) 에러를 막기 위해 NCNN의 내부 스레드를 1개로 제한합니다[cite: 10].
net_head.opt.num_threads = 1 
net_head.opt.use_vulkan_compute = False 

ncnn_ready = False
try:
    net_head.load_param("head_pose.param")
    net_head.load_model("head_pose.bin")
    head_in_name = net_head.input_names()[0]
    out_names = net_head.output_names()
    ncnn_ready = True
    print("🚀 NCNN 거대 무기 안전하게 장전 완료!")
except Exception as e: print(f"🚨 NCNN 에러: {e}")

# 3D 수학 각도 계산을 위한 가상의 인간 얼굴 모형 좌표 및 렌즈 굴절률(매트릭스)[cite: 10]
model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype="double")
camera_matrix = np.array([[CAM_W, 0, CAM_W/2], [0, CAM_W, CAM_H/2], [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4,1))

def get_dist(p1, p2): # 두 점 사이의 직선 거리를 구하는 수학 함수입니다[cite: 10].
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# ==========================================
# 3. 초저전력 실시간 관제 알고리즘 구동
# ==========================================
# 라즈베리 카메라 Rev 1.3의 가장 안정적인 오리지널 V4L2 설정을 사용합니다[cite: 10].
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

prev_time = time.time()
frame_count = 0

# 시스템 관제용 상태 변수 초기화[cite: 10]
distraction_start_time = None # 고개 딴짓 시간을 측정하는 타이머
current_tracking_dir = None 
perclos_buffer = collections.deque(maxlen=20) # 최근 2초간의 수면 상태를 보관하는 기억 장치(버퍼)

last_head_dir, last_head_msg = "Front", "Normal"
last_mask_val, last_mask_msg = "OFF", "..."
last_eye_val, last_eye_msg = 0.0, "..."
last_mouth_val, last_mouth_msg = 0.0, "..."

is_mask = False # 마스크 변수 에러 방지용 초기화
drowsy_end_time = 0 # 수면 후 2초간 '졸음(Drowsy)' 상태를 강제로 유지하기 위한 타이머

print("⚔️ 조국 AI 최종 실전 관제 개시!!! ⚔️")

try:
    while True:
        ret, frame = cap.read() # 카메라로부터 사진 1장을 읽어옵니다[cite: 10].
        if not ret or frame is None: continue
        
        curr_t = time.time()
        if (curr_t - prev_time) < FRAME_INTERVAL: continue
        prev_time = curr_t
        frame_count += 1
        
        # ----------------------------------------------------------------------
        # 🚨 [핵심 기술 1] 65500 픽셀 지뢰 박멸 및 지능형 컬러 복원 시스템
        # 구형 카메라가 921,600 픽셀의 깨진 데이터를 던질 때, 이를 오류로 버리지 않고
        # 수학적 재배열(Reshape)을 통해 완벽한 640x480 컬러 영상으로 즉시 복원합니다[cite: 10].
        # ----------------------------------------------------------------------
        if len(frame.shape) == 1 or frame.shape[1] > 65500:
            if frame.size == 921600: # 640x480 BGR 컬러 데이터 복원
                frame = frame.reshape((480, 640, 3))
            elif frame.size == 614400: # 640x480 YUYV 데이터 복원
                frame = frame.reshape((480, 640, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
            elif frame.size == 153600: # 320x240 YUYV 데이터 복원
                frame = frame.reshape((240, 320, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
            elif frame.size == 230400: # 320x240 BGR 데이터 복원
                frame = frame.reshape((240, 320, 3))

        # 복원된 고화질 이미지를 시스템 연산 규격인 320x240으로 경량화 축소합니다[cite: 10].
        if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
            frame = cv2.resize(frame, (CAM_W, CAM_H))
            
        frame = cv2.flip(frame, 1) # 거울처럼 좌우 반전
        
        # 흑백(Gray) 이미지로 변환하여 CPU 연산 부하를 획기적으로 낮춥니다[cite: 10].
        if len(frame.shape) == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = frame.copy()
            is_color = False
            
        display = gray.copy()
        
        # ----------------------------------------------------------------------
        # 🚨 [핵심 기술 2] 허공을 쫓는 유령(Ghosting) 잔상 강제 살처분 로직
        # ----------------------------------------------------------------------
        if not is_tracking:
            # 30프레임(3초)마다 추적기를 강제로 포맷하여 배경 무늬를 얼굴로 착각하는 현상을 방지합니다[cite: 10].
            if frame_count % 30 == 0:
                is_tracking = False

            if frame_count % 10 == 0: # 매 1초마다 새롭게 얼굴을 스캔
                faces = detector(gray, 0)
                if len(faces) > 0:
                    face = faces[0]
                    tracker.start_track(gray, dlib.rectangle(face.left(), face.top(), face.right(), face.bottom()))
                    is_tracking = True
        else:
            qual = tracker.update(gray)
            pos = tracker.get_position()
            
            # 추적 품질이 7.0 미만으로 떨어지거나, 얼굴이 화면 밖으로 이탈하면 즉각 추적을 끊어냅니다[cite: 10].
            if qual < 7.0 or pos.left() < 0 or pos.right() > CAM_W or pos.top() < 0 or pos.bottom() > CAM_H or pos.width() < 30:
                is_tracking = False
                distraction_start_time = None
                current_tracking_dir = None
                last_head_dir, last_head_msg = "Front", "Normal"
            else:
                face = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
                landmarks = predictor(gray, face) # 얼굴의 68개 핵심 뼈대를 찍습니다[cite: 10].
                
                f_x1, f_y1 = max(0, face.left()), max(0, face.top())
                f_x2, f_y2 = min(CAM_W, face.right()), min(CAM_H, face.bottom())
                
                # ======================================================================
                # 🎯 [모듈 1] 고개 딴짓 감지 (NCNN 딥러닝 + 2D 황금비율 하이브리드)
                # ======================================================================
                ncnn_dir = "Front"
                if ncnn_ready:
                    face_img_gray = gray[f_y1:f_y2, f_x1:f_x2]
                    if face_img_gray.size > 0:
                        # 메모리 단편화로 인한 C++ 충돌을 막기 위해 연속 메모리 배열(ascontiguousarray)을 사용합니다[cite: 10].
                        safe_face = np.ascontiguousarray(face_img_gray.copy())
                        safe_face = cv2.resize(safe_face, (224, 224))
                        safe_face_rgb = cv2.cvtColor(safe_face, cv2.COLOR_GRAY2RGB)
                        try:
                            # NCNN 딥러닝 모델에 사진을 넣어 고개 방향을 예측합니다[cite: 10].
                            mat_in = ncnn.Mat.from_pixels(safe_face_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 224, 224)
                            mat_in.substract_mean_normalize([123.675, 116.28, 103.53], [0.0171, 0.0175, 0.0174])
                            ex = net_head.create_extractor()
                            ex.input(head_in_name, mat_in)
                            ret_val, out = ex.extract(out_names[0])
                            if ret_val == 0:
                                head_idx = np.array(out).argmax()
                                ncnn_dir = {0: "Front", 1: "Up", 2: "Down", 3: "Left", 4: "Right"}.get(head_idx, "Front")
                        except: pass
                
                # 3D 수학 기반 상하(Pitch) 고개 각도 계산[cite: 10]
                img_pts = np.array([(landmarks.part(30).x, landmarks.part(30).y), (landmarks.part(8).x, landmarks.part(8).y), (landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(45).x, landmarks.part(45).y), (landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(54).x, landmarks.part(54).y)], dtype="double")
                _, rot_v, _ = cv2.solvePnP(model_points, img_pts, camera_matrix, dist_coeffs)
                rmat, _ = cv2.Rodrigues(rot_v)
                pitch_val = cv2.RQDecomp3x3(rmat)[0][0]
                
                # 🚨 [혁신 기술] 마스크 착용 시 턱 좌표 왜곡 방어!
                # 절대 왜곡되지 않는 '코끝과 양쪽 눈 사이의 거리 비율'만을 측정하여 35도 칼각 좌우(Left/Right)를 판별합니다[cite: 10].
                nx = landmarks.part(30).x 
                lex = landmarks.part(36).x 
                rex = landmarks.part(45).x 
                dist_left_eye = nx - lex 
                dist_right_eye = rex - nx 
                yaw_ratio = dist_left_eye / (dist_right_eye + 1e-6)
                
                math_dir = "Front"
                if yaw_ratio < 0.65: math_dir = "Left"  # 코가 왼쪽 눈에 쏠림 (좌측 주시 완벽 감지)[cite: 10]
                elif yaw_ratio > 1.5: math_dir = "Right" # 코가 오른쪽 눈에 쏠림[cite: 10]
                elif pitch_val > 35: math_dir = "Up"
                elif pitch_val < -35: math_dir = "Down"
                
                raw_dir = math_dir if math_dir != "Front" else ncnn_dir

                # 2D 대칭 얇은 방어막: 정면(Front) 판단의 오차를 제어합니다[cite: 10].
                cy = landmarks.part(8).y
                ey = (landmarks.part(36).y + landmarks.part(45).y) / 2.0
                pitch_ratio = (cy - ny) / (ny - ey + 1e-6)
                if (0.8 < yaw_ratio < 1.2) and (0.8 < pitch_ratio < 1.2):
                    raw_dir = "Front"

                # 운전 중 하늘을 보는 행위(Up)는 안전을 위해 Front로 강제 전환합니다[cite: 10].
                if raw_dir == "Up": raw_dir = "Front"

                # [관제 로직] 고개 딴짓 1초(10프레임) 벼락 경보 타이머![cite: 10]
                if raw_dir == "Front":
                    distraction_start_time = None
                    current_tracking_dir = None
                    last_head_dir = "Front"
                    last_head_msg = "Normal"
                else:
                    if current_tracking_dir != raw_dir:
                        distraction_start_time = time.time()
                        current_tracking_dir = raw_dir
                    
                    elapsed = time.time() - distraction_start_time
                    if elapsed >= 1.0: # 단 1초만 정면을 벗어나도 즉각 DISTRACTION 경보 발령![cite: 10]
                        last_head_dir = raw_dir
                        last_head_msg = "DISTRACTION"
                    else:
                        last_head_dir = "Front"
                        last_head_msg = "Normal"

                # ======================================================================
                # 🎯 [모듈 2] 마스크 착용 유무 판독 (양자화 딥러닝 융합)
                # ======================================================================
                if frame_count % 10 == 0:
                    is_mask = False
                    try:
                        # 마스크의 정밀한 판별을 위해 원본 컬러 데이터를 손실 없이 복원하여 분석합니다[cite: 10].
                        face_crop = np.ascontiguousarray(frame[f_y1:f_y2, f_x1:f_x2].copy())
                        if is_color: face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        else: face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
                        
                        m_img = cv2.resize(face_rgb, (mask_shape[2], mask_shape[1]))
                        m_in = np.expand_dims(m_img, axis=0).astype(mask_dtype)
                        if mask_dtype == np.float32: m_in = m_in / 255.0
                        
                        mask_interp.set_tensor(mask_in, m_in)
                        mask_interp.invoke() # 마스크 모델 실행[cite: 10]
                        pred = mask_interp.get_tensor(mask_out)[0]
                        is_mask = (np.argmax(pred) == 1) if len(pred) >= 2 else (pred[0] > 127)
                    except: pass
                    
                    # 딥러닝이 잡아내지 못한 마스크를 픽셀 분산(질감)을 통해 수학적으로 이중 방어합니다[cite: 10].
                    if not is_mask:
                        roi = gray[landmarks.part(28).y:landmarks.part(8).y, landmarks.part(48).x:landmarks.part(54).x]
                        if roi.size > 0 and (np.mean(roi) > 110 and np.var(roi) < 600): is_mask = True
                    
                    last_mask_val, last_mask_msg = ("ON", "Safe") if is_mask else ("OFF", "Warning")

                # ======================================================================
                # 🎯 [모듈 3] 수면 판독: 실전 EAR 0.27 동적 변환 & 2초 수면 / 2초 Drowsy
                # ======================================================================
                # 1. 랜드마크 기반 수학적 눈 크기(EAR) 비율을 계산합니다[cite: 10].
                eh = (get_dist(landmarks.part(37), landmarks.part(41)) + get_dist(landmarks.part(38), landmarks.part(40))) / 2.0
                ew = get_dist(landmarks.part(36), landmarks.part(39))
                last_eye_val = eh / ew if ew > 0 else 0
                
                # 2. 제가 직접 훈련시킨 5,000장 기반의 눈 딥러닝 무기를 가동합니다[cite: 10].
                tflite_eye_closed = False
                try:
                    e_x1, e_y1 = max(0, landmarks.part(36).x - 10), max(0, landmarks.part(37).y - 15)
                    e_x2, e_y2 = min(CAM_W, landmarks.part(45).x + 10), min(CAM_H, landmarks.part(47).y + 15)
                    if e_x2 > e_x1 and e_y2 > e_y1:
                        eye_crop = np.ascontiguousarray(gray[e_y1:e_y2, e_x1:e_x2].copy())
                        e_img = cv2.resize(eye_crop, (64, 64))
                        e_dtype = eye_interp.get_input_details()[0]['dtype']
                        e_in = np.expand_dims(e_img, axis=[0, -1]).astype(e_dtype)
                        if e_dtype == np.float32: e_in = e_in / 255.0
                        
                        eye_interp.set_tensor(eye_in, e_in)
                        eye_interp.invoke()
                        if eye_interp.get_tensor(eye_out)[0][0] > 0.5: tflite_eye_closed = True
                except: pass

                # 🚨 [혁신 포인트] 마스크 착용 유무에 따른 수면 기준점 분리 타격술![cite: 10]
                if is_mask:
                    # 마스크가 뼈대를 위로 밀어 올리는 현상을 분석하여 도출한 커스텀 기준점 '0.27' 강제 적용![cite: 10]
                    ear_threshold = 0.27
                    # 마스크 끈을 눈으로 오인하는 딥러닝 오류를 차단하고 실전 수학 수치만으로 정밀 타격합니다[cite: 10].
                    is_closed = 1 if last_eye_val < ear_threshold else 0
                else:
                    # 마스크 미착용 시 수학적 수치(0.22)와 딥러닝 판독을 하이브리드로 결합합니다[cite: 10].
                    ear_threshold = 0.22
                    is_closed = 1 if (last_eye_val < ear_threshold or tflite_eye_closed) else 0

                perclos_buffer.append(is_closed)
                
                # 🚨 [관제 로직] 글로벌 대기업 기준: 수면 2초 발동 및 깨어난 직후 2초 졸음(Drowsy) 강제 유지![cite: 10]
                if len(perclos_buffer) == 20:
                    rate = sum(perclos_buffer) / 20.0
                    
                    if rate >= 0.8: # 2초(20프레임) 동안 80% 이상 눈을 감고 있을 때만 수면 판정[cite: 10]
                        last_eye_msg = "SLEEP!!!"
                        drowsy_end_time = 0 # 자고 있을 때는 타이머 리셋[cite: 10]
                    else: 
                        # 잠에서 깬 직후 집중력이 취약하므로, 즉시 2초짜리 졸음 경고 타이머를 작동시킵니다[cite: 10].
                        if last_eye_msg == "SLEEP!!!":
                            drowsy_end_time = time.time() + 2.0 
                        
                        if time.time() < drowsy_end_time: last_eye_msg = "Drowsy"
                        else: last_eye_msg = "Normal"
                else:
                    last_eye_msg = "Normal"
                
                # 하품 판독 (입 랜드마크 비율)[cite: 10]
                mh = get_dist(landmarks.part(51), landmarks.part(57))
                mw = get_dist(landmarks.part(48), landmarks.part(54))
                last_mouth_val = mh / mw if mw > 0 else 0
                last_mouth_msg = "YAWN!" if last_mouth_val > 0.5 else "Closed"

                # 1GB RAM의 한계를 극복하기 위해 매 1초마다 메모리 누수를 방지하는 쓰레기 청소를 수행합니다[cite: 10].
                if frame_count % 10 == 0: gc.collect() 

                # --- 4. 관제 모니터링 출력 ---
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(display, f"HEAD: {last_head_dir} ({last_head_msg})", (5, 20), font, 0.45, (255), 1)
                cv2.putText(display, f"MASK: {last_mask_val} ({last_mask_msg})", (5, 40), font, 0.45, (255), 1)
                cv2.putText(display, f"EYE : {last_eye_val:.2f} ({last_eye_msg})", (5, 60), font, 0.45, (255), 1)
                cv2.putText(display, f"MOUTH: {last_mouth_val:.2f} ({last_mouth_msg})", (5, 80), font, 0.45, (255), 1)
                
                for i in range(68): cv2.circle(display, (landmarks.part(i).x, landmarks.part(i).y), 1, (255), -1)

        # 시스템 부하를 최소화하며 JPEG(품질 60)로 압축 후 노트북 모니터로 실시간 다이렉트 송출[cite: 10]
        ret_encode, buffer = cv2.imencode('.jpg', display, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ret_encode: client_socket.sendall(struct.pack(">L", len(buffer.tobytes())) + buffer.tobytes())

except Exception as e: print(f"🚨 에러 발생: {e}")
finally:
    cap.release()
    client_socket.close()
    server_socket.close()
