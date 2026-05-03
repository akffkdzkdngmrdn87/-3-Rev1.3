# 파일명: quantize_mask.py
# TFLite INT8 양자화(Quantization) 및 모델 경량화 스크립트

import os
import numpy as np
import cv2
import tensorflow as tf

# GPU 메모리 할당 오류 방지를 위한 CPU 연산 강제 할당 및 프로토콜 버퍼 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

print("메모리 제약 환경(1GB RAM) 최적화를 위한 INT8 양자화 프로세스 시작.")

# 1. 캘리브레이션을 위한 대표 데이터셋(Representative Dataset) 제너레이터 정의
# 양자화 과정에서 텐서의 활성화(Activation) 범위를 추정하기 위해 실제 이미지의 분포를 제공.
def representative_data_gen():
    img_dir = "dataset/0_nomask" # 샘플 데이터 디렉터리 경로
    img_names = os.listdir(img_dir)[:50] # 캘리브레이션용 샘플을 50장으로 제한하여 메모리 및 시간 효율성 확보
    for name in img_names:
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None: continue
        # 모델 입력 규격(224x224) 및 정규화(0~1 스케일링) 적용
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        yield [np.expand_dims(img, axis=0)]

# 2. Keras 원본 모델(.h5) 로드 및 TFLiteConverter 인스턴스화
try:
    model = tf.keras.models.load_model('mask_best.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
except Exception as e:
    print(f"[오류] 모델 로드 실패. 학습이 완료된 .h5 파일이 존재하는지 확인 요망: {e}")
    exit()

# 3. INT8 양자화 옵션 설정
converter.optimizations = [tf.lite.Optimize.DEFAULT] # 기본 최적화 옵션 활성화
converter.representative_dataset = representative_data_gen # 활성화 범위 추정을 위한 대표 데이터셋 설정

# 모델 내부의 연산(Ops)을 8비트 정수형(INT8)으로만 수행하도록 강제 지정
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # 모델의 입력 텐서 자료형을 정수형(uint8)으로 변환
converter.inference_output_type = tf.uint8 # 모델의 출력 텐서 자료형을 정수형(uint8)으로 변환

# 4. TFLite 변환 실행
tflite_quant_model = converter.convert()

# 5. 양자화된 최종 모델 파일(.tflite) 저장
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("INT8 양자화 완료. 최종 모델 파일 생성 성공: 'model_quantized.tflite'")

