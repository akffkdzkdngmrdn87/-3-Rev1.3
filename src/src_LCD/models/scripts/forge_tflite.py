# 파일명: forge_tflite.py
# Keras 모델 버전 호환성 패치 및 TFLite 변환 스크립트

import os
import json
import h5py

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

def fix_keras3_to_keras2(data):
    if isinstance(data, dict):
        if data.get("class_name") == "DTypePolicy":
            return data.get("config", {}).get("name", "float32")
        return {k: fix_keras3_to_keras2(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [fix_keras3_to_keras2(v) for v in data]
    else:
        return data

def patch_legacy_h5(h5_path):
    print(f"[{h5_path}] 모델 설정 속성(Attributes) 수정 진행 중...")
    with h5py.File(h5_path, 'r+') as f:
        if 'model_config' in f.attrs:
            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            
            config_str = config_str.replace('"batch_shape"', '"batch_input_shape"')
            
            try:
                config_json = json.loads(config_str)
                config_json = fix_keras3_to_keras2(config_json)
                config_str = json.dumps(config_json)
                f.attrs['model_config'] = config_str.encode('utf-8')
                print("Keras 3에서 Keras 2로의 하위 호환성 패치(DTypePolicy) 적용 완료.")
            except Exception as e:
                print(f"[예외 발생] 모델 속성 파싱 중 오류: {e}")

def convert_h5_to_tflite(h5_path, tflite_path):
    patch_legacy_h5(h5_path)
    print(f"[{h5_path}] TFLite 형식으로의 변환 프로세스 시작.")
    
    # Optimizer 가중치 로드 시 발생하는 예외를 방지하기 위해 
    # compile=False 옵션을 적용하여 추론(Inference) 전용 그래프만 추출.
    model = tf.keras.models.load_model(h5_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 모델 크기 최적화를 위한 TFLite 기본 양자화(Quantization) 옵션 적용.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite 변환 완료. 생성된 파일: [{tflite_path}]\n")

if __name__ == '__main__':
    print("=== TFLite 변환 파이프라인 가동 (TF 2.15 기준) ===")
    
    if os.path.exists("eye_best.h5"): convert_h5_to_tflite("eye_best.h5", "eye_best_quant.tflite")
    if os.path.exists("mouth_best.h5"): convert_h5_to_tflite("mouth_best.h5", "mouth_best_quant.tflite")
