
# 🛠️ Edge AI 모델 최적화 및 변환 스크립트 (Optimization Scripts)

본 `scripts/` 디렉터리는 데스크톱 환경에서 학습된 무거운 딥러닝 원본 모델(`.h5`, `.pth.tar`)을 Raspberry Pi 3 기반의 1GB RAM 엣지 환경에 맞게 극한으로 경량화하고 최적화하기 위해 구축된 파이썬(Python) 파이프라인 도구들을 포함하고 있습니다.

이 스크립트들은 엣지 디바이스 내부에서 실시간으로 실행되는 코드가 아니며, **모델 실전 배포 전(Pre-deployment) 단계에서 개발 PC 환경을 통해 수행되는 전처리 및 변환 도구**입니다.

---

## 1. 포함된 스크립트 명세 (Script Inventory)

### 1.1. TFLite 8비트 정수 양자화 파이프라인
*   **`quantize_mask.py`**
    *   **목적:** 마스크 착용 판별 원본 모델(`mask_best.h5`)을 INT8 규격으로 양자화(Quantization)합니다.
    *   **핵심 기술:** 대표 데이터셋(Representative Dataset) 캘리브레이션을 통해 입력부터 출력까지 모든 텐서(Tensor)를 `tf.uint8` 형식으로 강제 변환하여, 모델 크기를 75% 이상 압축하고 임베디드 환경에서의 연산 속도를 극대화합니다.
*   **`forge_tflite.py`**
    *   **목적:** 눈 깜박임(수면) 및 입 개폐(하품) 감지 Keras 원본 모델을 TFLite 포맷으로 변환합니다.
    *   **핵심 기술:** 최신 Keras 버전에서 발생할 수 있는 하위 호환성 문제(DTypePolicy)를 자체 패치하고, 불필요한 Optimizer 가중치를 배제(`compile=False` 적용)하여 순수 추론용(Inference) 그래프만 안정적으로 추출합니다.

### 1.2. PyTorch to ONNX 변환 파이프라인
*   **`pure_onnx_export.py`**
    *   **목적:** 고개 방향(Head Pose) 추정 모델인 PyTorch 체크포인트(`.pth.tar`)를 NCNN 모바일 프레임워크 컴파일의 전단계인 ONNX 포맷으로 변환합니다.
    *   **핵심 기술:** TensorFlow 등 타 프레임워크 라이브러리와의 메모리 충돌을 원천 차단하기 위해, 순수 PyTorch 네이티브 `export` 기능만을 격리 사용하여 가중치와 네트워크 구조를 온전히 추출합니다.
*   **`check_onnx_nodes.py`**
    *   **목적:** 변환이 완료된 ONNX 모델의 무결성을 시스템적으로 검증합니다.
    *   **핵심 기술:** 무거운 가중치 데이터를 제외하고 모델의 그래프(Graph) 구조만 메모리에 로드하여, 입력(Input) 및 출력(Output) 노드의 형태(Shape)가 타겟 프레임워크 요구사항에 부합하게 정상적으로 추출되었는지 신속하게 검사합니다.

---

## 2. 구동 환경 및 주의 사항 (Prerequisites)
*   본 디렉터리의 스크립트들을 실행하기 위해서는 TensorFlow 2.x, PyTorch, ONNX, OpenCV 등의 머신러닝 라이브러리가 구성된 호스트 개발 환경이 요구됩니다.
*   본 스크립트들을 통해 성공적으로 경량화 및 변환이 완료된 최종 산출물(`.tflite`, `.onnx`, `.bin`, `.param`)만을 상위 디렉터리인 `models/` 에 물리적으로 배치하여 실제 엣지 디바이스(Raspberry Pi) 시스템에서 운용합니다.
