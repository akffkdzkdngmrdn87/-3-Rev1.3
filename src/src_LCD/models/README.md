# 🧠 Edge AI 모델 저장소 및 양자화(Quantization) 최적화 명세서

본 `models/` 디렉터리는 Raspberry Pi 3 Model B의 1GB RAM 및 제한된 연산 자원 환경에서 실시간(Zero-Latency) 운전자 상태 모니터링을 수행하기 위해 최적화된 경량 딥러닝 모델들을 포함하고 있습니다. 

엣지 디바이스(Edge Device) 배포를 목적으로 아키텍처가 설계되었으므로, 용량이 큰 원본 학습 모델(.h5, .pth.tar)은 저장소에서 제외하였으며 연산 효율을 위해 압축된 최종 배포용(Deployment) 모델만 적재되어 있습니다.

---

## 1. 탑재된 인공지능 모델 구성 (Model Inventory)

본 저장소는 다중 상태 독립 감지를 위해 다음과 같은 개별 경량화 모델들로 구성되어 있습니다.

*   **`shape_predictor_68_face_landmarks.dat`**: Dlib 기반 68개 안면 랜드마크 추출기 (기하학적 수식 연산의 기준점 제공)
*   **`eye_best_quant.tflite`**: 안검 폐쇄(수면) 확률 연산 모델 (TensorFlow Lite 8-bit 양자화)
*   **`mouth_best_quant.tflite`**: 입 개폐(하품) 확률 연산 모델 (TensorFlow Lite 8-bit 양자화)
*   **`model_quantized.tflite`**: 마스크 착용 유무 판별 모델 (TensorFlow Lite 8-bit 양자화)
*   **`head_pose.bin` & `head_pose.param`**: 고개 방향(시선 이탈) 탐지 모델 (NCNN 프레임워크 최적화)

---

## 2. 학습 데이터 출처 및 라이선스 (Training Dataset Reference)

본 시스템의 안검 개폐 및 하품 등 주요 운전자 상태 판독 모델은 공공 데이터를 기반으로 전처리 및 미세 조정(Fine-tuning) 되었습니다. 모델의 학습 저작권 및 출처는 아래와 같습니다.

*   **데이터셋 명칭:** AI 허브(AI Hub) - 운전자 상태 인지 데이터
*   **데이터 출처:** [AI Hub - 졸음운전 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?srchOptnCnd=OPTNCND001&currMenu=115&topMenu=100&searchKeyword=%EC%A1%B8%EC%9D%8C%EC%9A%B4%EC%A0%84&aihubDataSe=data&dataSetSn=173)
*   **활용 내역:** 한국인 운전자의 실제 주행 환경 및 조명 변화가 반영된 데이터를 선별 추출하여, 모델의 실전 인식률(Robustness)과 일반화 성능을 극대화하는 데 사용하였습니다.

---

## 3. 모델 경량화 및 최적화 메커니즘 (Edge Optimization Process)

본 프로젝트는 GPU 기반 서버급 연산 자원이 없는 초소형 임베디드 보드에서의 구동을 위해, 자체 파이썬 스크립트 파이프라인을 구축하여 원본 신경망을 극단적으로 경량화했습니다.

### 3.1. 8비트 정수 극한 양자화 (INT8 Quantization)
TensorFlow의 `TFLiteConverter`를 활용하여 원본 신경망 모델의 32비트 부동소수점(Float32) 가중치를 8비트 정수형(INT8)으로 압축하였습니다. 
*   **적용 파이프라인 스크립트:** `quantize_mask.py`, `forge_tflite.py`
*   **최적화 세부 사항:** 대표 데이터셋(Representative Dataset) 캘리브레이션 프로세스를 스크립트에 포함하여, 모델의 입출력 텐서 타입까지 완벽한 `tf.uint8`로 제한하는 극한의 최적화를 수행했습니다. 또한 호환성 충돌을 방지하기 위해 Optimizer 요소를 배제하는 자체 파싱 로직을 적용했습니다.
*   **결과:** 이를 통해 모델 용량을 약 75% 감소시키고, Raspberry Pi 3의 메모리 캐시 적중률을 높여 추론 지연 시간(Latency)을 대폭 단축했습니다.

### 3.2. 모바일 최적화 프레임워크 전환 (PyTorch -> ONNX -> NCNN)
고도화된 연산이 요구되는 Head Pose(고개 방향) 추정 모델은 Tencent의 모바일 및 엣지 전용 프레임워크인 NCNN 포맷(`.bin`, `.param`)으로 변환하여 ARM Cortex-A53 CPU 아키텍처에서의 연산 효율성을 극대화했습니다. 
*   **적용 파이프라인 스크립트:** `pure_onnx_export.py`, `check_onnx_nodes.py`
*   **최적화 세부 사항:** 프레임워크 간의 종속성을 배제하기 위해 텐서플로우 개입 없이 PyTorch 모델(`.pth.tar`)을 순수 ONNX 규격으로 추출하고, 입출력 노드의 형태(Shape) 무결성을 검증하는 자체 파이프라인을 구축했습니다.
*   **결과:** 다중 프레임워크 혼용 환경에서 발생할 수 있는 텐서 노드(Tensor Node) 충돌을 수학적으로 방어하고, 엣지 환경에 완벽히 호환되는 초경량 NCNN 컴파일을 성공적으로 수행했습니다.

### 3.4. 메모리 제약 극복 전략
학습 과정의 원본 데이터(HDF5, PyTorch 체크포인트) 및 파이썬 최적화 스크립트를 엣지 기기 배포본에서 제외하고, 상기 서술된 최종 경량화 컴파일 파일만 물리적으로 배치함으로써 디스크 I/O 병목 현상을 방지하고 시스템 기동 속도를 극대화했습니다.

* 기타 스크립트 파일들은 이 폴더 하위 폴더인 scripts/에 있습니다.
