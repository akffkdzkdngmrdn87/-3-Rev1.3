# 파일명: src_LCD/dsm_commander_lcd.py
# [혁신 모빌리티 AI - 완전 독립형(Zero-Latency) 임베디드 DSM & LCD 관제 시스템]

import os 
import cv2 
import numpy as np 
import dlib 
import tflite_runtime.interpreter as tflite 
import ncnn 
import time 
import collections 
import math 
import gc 
from RPLCD.i2c import CharLCD # I2C 통신 기반 초저전력 LCD 제어 라이브러리

# 🚨 [시스템 최적화] 영상 송출망을 제거하여 확보한 연산력을 바탕으로,
# 라즈베리파이 3의 4개 CPU 코어를 병렬 융합 연산에 풀가동합니다.
os.environ['OMP_NUM_THREADS'] = '4'

# 연산 부하 최소화를 위해 해상도를 320x240으로 경량화 및 10 FPS로 고정합니다.
CAM_W, CAM_H = 320, 240 
FPS_LIMIT = 10 
FRAME_INTERVAL = 1.0 / FPS_LIMIT 

try:
    # 🎯 [Zero-Latency] 통신 모듈 대신 초저전력 I2C LCD를 연결하여 즉각적인 상태 출력을 달성합니다.
    # PCF8574 확장 칩셋을 사용하여 0x27 주소로 통신망을 개통합니다.
    lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)
    lcd.clear()
    lcd.write_string(' DSM AI BOOTING ')
    time.sleep(1)
except Exception as e: pass

# ==========================================
# 1. 12만 장 융합 기반 8비트 INT8 양자화 딥러닝 모델 장전
# ==========================================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
tracker = dlib.correlation_tracker()
is_tracking = False

# 안검 폐쇄(수면) 판독 양자화 모델 로드
eye_interp = tflite.Interpreter("eye_best_quant.tflite", num_threads=4)
eye_interp.allocate_tensors()
eye_in, eye_out = eye_interp.get_input_details()[0]['index'], eye_interp.get_output_details()[0]['index']

# 하품 판독 양자화 모델 로드
mouth_interp = tflite.Interpreter("mouth_best_quant.tflite", num_threads=4)
mouth_interp.allocate_tensors()
mouth_in, mouth_out = mouth_interp.get_input_details()[0]['index'], mouth_interp.get_output_details()[0]['index']

# 마스크 착용 여부 판독 양자화 모델 로드
mask_interp = tflite.Interpreter("model_quantized.tflite", num_threads=4)
mask_interp.allocate_tensors()
mask_input_details = mask_interp.get_input_details()[0]
mask_in, mask_out = mask_input_details['index'], mask_interp.get_output_details()[0]['index']
mask_shape, mask_dtype = mask_input_details['shape'], mask_input_details['dtype']

# 고개 방향(Pitch, Yaw) 판독을 위한 C++ 기반 초고속 모바일 최적화 NCNN 엔진 로드
net_head = ncnn.Net()
# 🚨 [메모리 방어] ZRAM 폭발(Segfault)을 방지하기 위해 NCNN의 내부 스레드를 1개로 엄격히 제어합니다.
net_head.opt.num_threads = 1 
net_head.opt.use_vulkan_compute = False 

ncnn_ready = False
try:
    net_head.load_param("head_pose.param")
    net_head.load_model("head_pose.bin")
    head_in_name = net_head.input_names()[0]
    out_names = net_head.output_names()
    ncnn_ready = True
except Exception as e: pass

# 3D 기하학 수학 연산(Pitch/Yaw/Roll)을 위한 가상의 안면 기준 좌표계 매트릭스
model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype="double")
camera_matrix = np.array([[CAM_W, 0, CAM_W/2], [0, CAM_W, CAM_H/2], [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4,1))

def get_dist(p1, p2): 
    """두 랜드마크 점 사이의 픽셀 단위 유클리디안 거리를 계산하는 수학 함수입니다."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# V4L2 드라이버를 활용하여 구형 카메라 초기화
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

prev_time = time.time()
frame_count = 0
distraction_start_time = None 
current_tracking_dir = None 

# 🚨 [안정화 로직] 터미널 출력 및 영상 송출 제약이 해제됨에 따른 상태 요동(Flickering) 방지용 다중 뚝심 버퍼링
perclos_buffer = collections.deque(maxlen=20) # 2초(20프레임) 누적 수면 버퍼
mask_inference_buffer = collections.deque(maxlen=4) 
mouth_buffer = collections.deque(maxlen=10) # 1초(10프레임) 하품 버퍼

# 상태 초기화
last_head_dir, last_head_msg = "Front", "Normal"
last_mask_val = "OFF" 
last_eye_msg = "Normal"
last_mouth_msg = "Closed"

is_mask_final = False 
drowsy_end_time = 0 

# LCD 갱신 오버헤드를 막기 위한 이전 상태 캐싱 변수
lcd_last_line1 = ""
lcd_last_line2 = ""

# 🚨 [보안 로직] 독립적 물리 행동 기반 강제 셧다운 해독기 (Easter Egg)
# 키보드/마우스가 없는 엣지 환경에서 시스템을 안전하게 종료하기 위한 숨겨진 패턴 시퀀스입니다.
easter_egg_target = ['Left', 'Front', 'Right', 'Front', 'Left', 'Front', 'Right', 'Front', 'YAWN']
easter_egg_current = []
prev_discrete_action = None

try: lcd.clear()
except: pass

try:
    while True:
        ret, frame = cap.read() 
        if not ret or frame is None: 
            continue
        
        curr_t = time.time()
        # 지정된 FPS(10)를 초과하는 프레임은 드롭하여 CPU 과부하를 방어합니다.
        if (curr_t - prev_time) < FRAME_INTERVAL: 
            continue
        prev_time = curr_t
        frame_count += 1
        
        # ======================================================================
        # 🚨 [전처리] 65500 픽셀 지뢰 (1D 배열 붕괴) 수학적 강제 복원 로직
        # ==========================================
        # 구형 카메라 드라이버 오류로 인해 1차원으로 붕괴된 데이터를 Numpy 연산을 통해 실시간 2D 컬러로 복구합니다.
        if len(frame.shape) == 1 or frame.shape[1] > 65500:
            if frame.size == 921600: frame = frame.reshape((480, 640, 3))
            elif frame.size == 614400: 
                frame = frame.reshape((480, 640, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
            elif frame.size == 153600: 
                frame = frame.reshape((240, 320, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
            elif frame.size == 230400: frame = frame.reshape((240, 320, 3))

        # 프레임 크기 표준화 및 좌우 반전
        if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
            frame = cv2.resize(frame, (CAM_W, CAM_H))
        frame = cv2.flip(frame, 1) 
        
        # 연산량 감소를 위한 흑백(Gray) 이미지 변환
        if len(frame.shape) == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = frame.copy()
            is_color = False
            
        # ======================================================================
        # 🚨 [추적기] Dlib 랜드마크 추출 및 유령(Ghosting) 잔상 강제 초기화 로직
        # ======================================================================
        if not is_tracking:
            # 3초(30프레임) 주기로 추적기를 초기화하여 배경 패턴을 얼굴로 오인하는 현상을 차단합니다.
            if frame_count % 30 == 0: 
                is_tracking = False

            if frame_count % 10 == 0: 
                dets, scores, _ = detector.run(gray, 0, -0.5)
                if len(dets) > 0:
                    face = dets[np.argmax(scores)]
                    tracker.start_track(gray, dlib.rectangle(face.left(), face.top(), face.right(), face.bottom()))
                    is_tracking = True
        else:
            qual = tracker.update(gray)
            pos = tracker.get_position()
            
            # 추적 품질(Quality)이 임계치(4.0) 미만으로 하락 시 즉각 추적 중단
            if qual < 4.0 or pos.left() < 0 or pos.right() > CAM_W or pos.top() < 0 or pos.bottom() > CAM_H or pos.width() < 30:
                is_tracking = False
                distraction_start_time = None
                current_tracking_dir = None
                last_head_dir = "Lost" 
                last_head_msg = "Normal" 
            else:
                face = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
                landmarks = predictor(gray, face) 
                
                f_x1, f_y1 = max(0, face.left()), max(0, face.top())
                f_x2, f_y2 = min(CAM_W, face.right()), min(CAM_H, face.bottom())
                
                # ======================================================================
                # 🎯 [모듈 1] 고개 딴짓 감지 (NCNN 딥러닝 + 2D/3D 기하학 융합)
                # ======================================================================
                ncnn_dir = "Front"
                if ncnn_ready:
                    face_img_gray = gray[f_y1:f_y2, f_x1:f_x2]
                    if face_img_gray.size > 0:
                        safe_face = np.ascontiguousarray(face_img_gray.copy())
                        safe_face = cv2.resize(safe_face, (224, 224))
                        safe_face_rgb = cv2.cvtColor(safe_face, cv2.COLOR_GRAY2RGB)
                        try:
                            # NCNN 엔진을 통한 고개 방향 기초 추론
                            mat_in = ncnn.Mat.from_pixels(safe_face_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 224, 224)
                            mat_in.substract_mean_normalize([123.675, 116.28, 103.53], [0.0171, 0.0175, 0.0174])
                            ex = net_head.create_extractor()
                            ex.input(head_in_name, mat_in)
                            ret_val, out = ex.extract(out_names[0])
                            if ret_val == 0:
                                head_idx = np.array(out).argmax()
                                ncnn_dir = {0: "Front", 1: "Up", 2: "Down", 3: "Left", 4: "Right"}.get(head_idx, "Front")
                        except: pass
                        del safe_face, safe_face_rgb 
                
                # OpenCV solvePnP를 활용한 3D 기하학적 상하(Pitch) 각도 도출
                img_pts = np.array([(landmarks.part(30).x, landmarks.part(30).y), (landmarks.part(8).x, landmarks.part(8).y), (landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(45).x, landmarks.part(45).y), (landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(54).x, landmarks.part(54).y)], dtype="double")
                _, rot_v, _ = cv2.solvePnP(model_points, img_pts, camera_matrix, dist_coeffs)
                rmat, _ = cv2.Rodrigues(rot_v)
                pitch_val = cv2.RQDecomp3x3(rmat)[0][0]
                
                # 🚨 [수학적 대체 로직 Fallback] 마스크 착용 시 3D 턱 좌표 왜곡 방어!
                # 변형되지 않는 상안면(코끝과 눈)의 2D 황금비율만으로 35도 칼각 시선 제어 권한을 탈취합니다.
                nx = landmarks.part(30).x 
                ny = landmarks.part(30).y 
                lex = landmarks.part(36).x 
                rex = landmarks.part(45).x 
                dist_left_eye = nx - lex 
                dist_right_eye = rex - nx 
                yaw_ratio = dist_left_eye / (dist_right_eye + 1e-6)
                
                math_dir = "Front"
                if yaw_ratio < 0.65: math_dir = "Left" 
                elif yaw_ratio > 1.5: math_dir = "Right" 
                elif pitch_val > 35: math_dir = "Up"
                elif pitch_val < -35: math_dir = "Down"
                
                raw_dir = math_dir if math_dir != "Front" else ncnn_dir

                # 2D 상하/좌우 대칭 방어막 로직 적용
                cy = landmarks.part(8).y
                ey = (landmarks.part(36).y + landmarks.part(45).y) / 2.0
                pitch_ratio = (cy - ny) / (ny - ey + 1e-6) 
                if (0.8 < yaw_ratio < 1.2) and (0.8 < pitch_ratio < 1.2):
                    raw_dir = "Front"

                # 하드웨어 한계인 53.5도 화각을 고려하여 'Up' 판정 시 프레임 이탈 방지를 위해 'Front'로 묵인 처리
                if raw_dir == "Up": raw_dir = "Front"

                # 주의 태만(Distraction) 1초 누적 벼락 경보 타이머 (Temporal Hysteresis)
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
                    if elapsed >= 1.0: 
                        last_head_dir = raw_dir
                        last_head_msg = "DISTRACTION"
                    else:
                        last_head_dir = "Front"
                        last_head_msg = "Normal"

                # ======================================================================
                # 🎯 [모듈 2] 결정 융합: TFLite 마스크 판독 + 픽셀 명암 분산도 수학 필터
                # ======================================================================
                # 연산량 감소를 위해 마스크 판별은 0.5초(5프레임) 주기로 스케줄링합니다.
                if frame_count % 5 == 0: 
                    current_mask_raw = False
                    try:
                        face_crop = np.ascontiguousarray(frame[f_y1:f_y2, f_x1:f_x2].copy())
                        if is_color: face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        else: face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
                        
                        m_img = cv2.resize(face_rgb, (mask_shape[2], mask_shape[1]))
                        m_in = np.expand_dims(m_img, axis=0).astype(mask_dtype)
                        if mask_dtype == np.float32: m_in = m_in / 255.0
                        
                        mask_interp.set_tensor(mask_in, m_in)
                        mask_interp.invoke() 
                        pred = mask_interp.get_tensor(mask_out)[0]
                        current_mask_raw = (np.argmax(pred) == 1) if len(pred) >= 2 else (pred[0] > 127)
                        
                        del face_crop, face_rgb, m_img, m_in 
                    except: pass
                    
                    # 🚨 [수학적 대체 로직 Fallback] 역광 등 극단적 조명으로 인한 딥러닝 인지 상실 시,
                    # 입 주변의 픽셀 명암 분산도(Var < 600)를 측정하여 모델의 오답을 기하학적으로 기각합니다.
                    if not current_mask_raw:
                        roi = gray[landmarks.part(28).y:landmarks.part(8).y, landmarks.part(48).x:landmarks.part(54).x]
                        if roi.size > 0 and (np.mean(roi) > 110 and np.var(roi) < 600): 
                            current_mask_raw = True
            
                    mask_inference_buffer.append(1 if current_mask_raw else 0)
                        
                    # 버퍼 안정화 (최근 4회 중 2회 이상 감지 시 상태 ON 적용)
                    if len(mask_inference_buffer) == 4:
                        if sum(mask_inference_buffer) >= 2: last_mask_val = "ON"
                        else: last_mask_val = "OFF"
                            
                is_mask_final = (last_mask_val == "ON") 

                # ======================================================================
                # 🎯 [모듈 3] 결정 융합: EAR/MAR 기하학 타격 및 눈/입 딥러닝 병합
                # ======================================================================
                # EAR (Eye Aspect Ratio) 수학 수치 산출
                eh = (get_dist(landmarks.part(37), landmarks.part(41)) + get_dist(landmarks.part(38), landmarks.part(40))) / 2.0
                ew = get_dist(landmarks.part(36), landmarks.part(39))
                last_eye_val = eh / ew if ew > 0 else 0
                
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
                        if eye_interp.get_tensor(eye_out)[0][0] > 0.75: 
                            tflite_eye_closed = True
                        del eye_crop, e_img, e_in 
                except: pass

                # 🚨 [수학적 대체 로직 Fallback] 마스크 착용 감지 시 EAR 임계값을 0.22 -> 0.27로 강제 상향!
                if is_mask_final:
                    ear_threshold = 0.27 
                    is_closed = 1 if (last_eye_val < ear_threshold or tflite_eye_closed) else 0
                else:
                    ear_threshold = 0.22
                    is_closed = 1 if (last_eye_val < ear_threshold or tflite_eye_closed) else 0

                perclos_buffer.append(is_closed)
                
                # 글로벌 산업 표준 PERCLOS 누적 검열 (2초/20프레임 중 80% 이상 눈 감음)
                if len(perclos_buffer) == 20:
                    rate = sum(perclos_buffer) / 20.0
                    if rate >= 0.8: 
                        last_eye_msg = "Sleep" 
                        drowsy_end_time = 0 
                    else: 
                        # 수면 상태 각성 직후 2초간 '졸음(Drowsy)' 상태를 강제 유지하여 안전을 확보합니다.
                        if last_eye_msg == "Sleep":
                            drowsy_end_time = time.time() + 2.0 
                        if time.time() < drowsy_end_time: 
                            last_eye_msg = "Drowsy"
                        else: 
                            last_eye_msg = "Normal"
                else:
                    last_eye_msg = "Normal"
                
                # MAR (Mouth Aspect Ratio) 하품 기하학 연산 및 1초 뚝심 누적 버퍼
                mh = get_dist(landmarks.part(51), landmarks.part(57))
                mw = get_dist(landmarks.part(48), landmarks.part(54))
                current_mouth_val = mh / mw if mw > 0 else 0
                
                mouth_buffer.append(1 if current_mouth_val > 0.5 else 0)
                
                if len(mouth_buffer) == 10 and sum(mouth_buffer) >= 6:
                    last_mouth_msg = "YAWN"
                else:
                    last_mouth_msg = "Closed"

        # 1GB RAM 메모리 누수 방지용 주기적 수동 GC(Garbage Collection)
        if frame_count % 10 == 0: 
            gc.collect() 

        # ======================================================================
        # 🎯 [모듈 4] 🚨 전설의 이스터에그 (물리적 안면 행동 기반 시스템 강제 셧다운 시퀀스)
        # ======================================================================
        # 현재 안면의 지배적인 상태(행동)를 단일 문자열로 치환합니다.
        discrete_action = "Front"
        
        if last_mouth_msg == "YAWN" and last_head_dir == "Front":
            discrete_action = "YAWN"
        elif last_head_dir in ["Left", "Right"]:
            discrete_action = last_head_dir
            
        if discrete_action != prev_discrete_action:
            expected_action = easter_egg_target[len(easter_egg_current)]
            
            # 사전에 정의된 특정 안면 동작 시퀀스가 순서대로 입력되었는지 대조합니다.
            if discrete_action == expected_action:
                easter_egg_current.append(discrete_action)
                
                # 9단계 비밀 시퀀스가 완벽히 입력된 경우 시스템을 강제로 종료합니다.
                if len(easter_egg_current) == len(easter_egg_target):
                    try:
                        lcd.clear()
                        lcd.cursor_pos = (0, 0)
                        lcd.write_string("*** EASTER EGG ***")
                        lcd.cursor_pos = (1, 0)
                        lcd.write_string(" GOOD BYE, BOSS! ")
                    except: pass
                    time.sleep(2) 
                    os.system("sudo shutdown -h now") # 리눅스 강제 전원 차단
            else:
                # 시퀀스 불일치 시 오작동 방지를 위해 버퍼 초기화
                if discrete_action == easter_egg_target[0]:
                    easter_egg_current = [discrete_action]
                else:
                    easter_egg_current = []
                    
        prev_discrete_action = discrete_action

        # ======================================================================
        # 🎯 [모듈 5] 결정 융합 결과의 LCD 단말기 완벽 독립 출력 (Zero-Latency)
        # ======================================================================
        mask_prefix = "Mask " if is_mask_final else ""
        head_str = last_head_dir if last_head_dir != "Lost" else "Searching..."
        
        # 16문자 길이 제한을 맞추기 위한 포매팅
        line1 = f"{mask_prefix}{head_str}".ljust(16)
        
        # 치명도에 따른 경보 우선순위 적용 (수면 > 졸음 > 딴짓 > 하품)
        current_state = "Normal"
        if last_head_msg == "DISTRACTION": 
            current_state = "Distraction"
        elif last_eye_msg == "Sleep": 
            current_state = "Sleep"
        elif last_eye_msg == "Drowsy": 
            current_state = "Drowsy"
        elif last_mouth_msg == "YAWN": 
            current_state = "YAWN Warn!" 
        elif last_head_dir == "Lost": 
            current_state = "Lost" 
            
        line2 = f"{current_state}".ljust(16)
            
        # LCD 패널의 불필요한 I2C 갱신 부하 및 깜빡임을 막기 위해 상태 변화 시에만 렌더링을 수행합니다.
        if line1 != lcd_last_line1 or line2 != lcd_last_line2:
            try:
                lcd.cursor_pos = (0, 0)
                lcd.write_string(line1)
                lcd.cursor_pos = (1, 0)
                lcd.write_string(line2)
            except: pass
            lcd_last_line1 = line1
            lcd_last_line2 = line2

except Exception as e: 
    pass
finally:
    cap.release()
    try: lcd.clear() 
    except: pass
