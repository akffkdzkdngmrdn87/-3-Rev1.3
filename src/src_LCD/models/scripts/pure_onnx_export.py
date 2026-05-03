# 파일명: pure_onnx_export.py
# PyTorch 가중치(.pth.tar)를 ONNX 포맷으로 직접 추출하는 스크립트

import torch
import torch.nn as nn
from torchvision import models
import os

# TensorFlow 기반 모듈과의 충돌(Core Dump)을 방지하기 위해 PyTorch 네이티브 export 기능만 사용.

def export_to_onnx(ckpt_path, onnx_name, num_classes):
    print(f"[{ckpt_path}] 모델을 [{onnx_name}] (분류 클래스 수: {num_classes}) 형식으로 변환 시작.")
    
    if not os.path.exists(ckpt_path):
        print(f"[오류] 지정된 경로에 체크포인트 파일이 존재하지 않습니다: {ckpt_path}")
        return

    # 1. ResNet50 기본 아키텍처 인스턴스화
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    
    # 목적에 맞는 최종 출력 레이어(Fully Connected Layer) 차원 수정.
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 2. 학습된 체크포인트 가중치 로드 (CPU 환경 매핑)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    
    # DataParallel 등으로 학습된 모델의 'module.' 접두사 제거를 통한 State Dict 호환성 확보
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("가중치 로드 및 평가(Eval) 모드 전환 완료.")

    # 3. 변환을 위한 더미 입력 텐서 생성 (입력 크기: 1x3x224x224) 및 ONNX 추출
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_name, 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['output'])
                      
    print(f"ONNX 모델 추출 성공: [{onnx_name}]\n")

if __name__ == '__main__':
    print("=== PyTorch to ONNX 변환 파이프라인 가동 ===")
    
    # Head Pose(고개 방향 추정) 모델 변환: 5개 클래스 분류
    export_to_onnx("checkpoint-8.pth.tar", "head_pose.onnx", 5)
    
    # Drowsiness(눈 깜박임 상태) 모델 변환: 2개 클래스 분류
    export_to_onnx("checkpoint-19.pth.tar", "drowsy_eyes.onnx", 2)
