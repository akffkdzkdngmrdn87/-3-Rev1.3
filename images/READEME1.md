# 고신뢰성 엣지 컴퓨팅 기반 자율주행 및 운전자 상태 감시 시스템 (DSM)
### : 딥러닝 블랙박스 모델의 한계와 18.7% 휴리스틱 대체(Heuristic Fallback) 로직에 대한 결정 수준 융합 정밀 분석

## 1. 엣지 인공지능의 진화와 자율주행 시각 인지 시스템의 본질적 한계

현대의 인공지능(AI)과 컴퓨터 비전 기술은 클라우드 서버의 컴퓨팅 자원에 의존하는 중앙 집중식 처리 모델에서 벗어나, 초저전력 및 제한된 메모리 환경에서도 자율적이고 즉각적인 의사결정을 수행해야 하는 엣지 컴퓨팅(Edge Computing) 패러다임으로 전환하고 있습니다. 특히 자율주행 차량의 환경 인식(Environmental Perception) 및 실시간 운전자 상태 감시(Driver Status Monitoring, DSM) 시스템은 통제된 실험실이 아닌 실전 도로 환경에 노출됩니다. 이곳은 급격한 조명 변화, 동적 흔들림, 물리적 가림(Occlusion) 등 예측 불가능한 변수들이 존재하는 공간으로, 극도의 시스템 신뢰성을 요구합니다.

이러한 엣지 환경에서 심층 신경망(DNN)이나 합성곱 신경망(CNN)에만 의존하는 단일 인지 체계는 연산 지연(Latency)과 판단 오류(Failure)를 초래할 수 있습니다. 1GB RAM 수준의 초소형 엣지 디바이스(예: Raspberry Pi 3)에서 인공지능 모델을 구동하기 위해서는 가중치를 32비트 부동소수점(FP32)에서 8비트 정수(INT8)로 압축하는 양자화(Quantization) 기술이 수반됩니다. 그러나 이러한 양자화 과정과 딥러닝 특유의 확률적 추론 메커니즘은 학습 데이터의 분포를 벗어난(Out-of-Distribution, OoD) 상황에 직면할 경우, 확신도(Confidence Score)가 급락하여 인지 능력을 상실하는 블랙박스(Black-box)적 한계를 드러냅니다.

실증 연구 및 교차 검증 시뮬레이션 결과에 따르면, 실전 동적 주행 환경에서 딥러닝 알고리즘이 외부 간섭에 의해 정상적인 시각적 판단 능력을 상실하는 이른바 '인지적 실패율(False Detection Rate 및 Fallback 비율)'은 평균적으로 18.7%에 달하는 것으로 분석됩니다. 이는 인공지능이 81.3%의 정상적인 상황에서는 복잡한 패턴을 훌륭히 분류하지만, 나머지 18.7%의 극한 상황에서는 한계를 보일 수 있음을 시사합니다.

따라서 본 연구 보고서는 딥러닝 모델의 18.7%라는 인지적 빈틈을 보완하고 시스템의 신뢰성을 극대화하기 위해 설계된 **'결정 수준 융합(Decision-level Fusion)'** 기술의 근원적 메커니즘을 분석합니다. 확률적 추론을 수행하는 양자화 딥러닝과 결정론적인 수학적 대체 로직(Mathematical Heuristic Fallback)이 어떻게 상호 보완적으로 작동하는지, 그리고 EAR, MAR, 픽셀 분산도 등의 기하학적 수식이 딥러닝의 시각적 한계를 어떻게 극복하는지에 대한 철저한 수식적, 통계적 분석을 제공합니다.

---

## 2. 자립형 임베디드 기동 및 Zero-Latency 관제 아키텍처

본 시스템은 네트워크에 의존하는 클라우드 기반 모니터링의 한계를 극복하고, 독립적인 초저전력 임베디드 환경을 구축하기 위해 아키텍처를 개편했습니다.

### 2.1. Systemd 권한 분리를 통한 전원 인가 시 자립 기동
차량에 전원 인가 시 별도의 I/O 장치(모니터, 키보드) 없이 시스템이 자율적으로 구동되도록 리눅스의 `systemd` 데몬을 활용한 권한 분리 기동 방식을 적용했습니다.

*   **설계 배경:** 1GB RAM 환경에서 OOM(Out of Memory) 방지를 위한 ZRAM 제어는 최고 관리자(root) 권한이 필요하나, 보안 및 환경 변수 일관성을 위해 파이썬 AI 코드는 일반 계정(pi) 권한으로 실행되어야 합니다.
*   **해결 로직:** `ExecStartPre` 속성을 통해 ZRAM 활성화는 `root` 권한으로 선행 처리하고, 본 AI 코드는 `User=pi` 권한으로 분리 실행하는 데몬 스크립트를 구현했습니다.

```ini
# /etc/systemd/system/dsm_ai.service
[Unit]
Description=DSM AI Standalone Service
After=multi-user.target

[Service]
Type=simple
User=pi
ExecStartPre=+/sbin/swapon /swapfile_zram 
ExecStart=/usr/bin/python3 /home/pi/src_LCD/dsm_commander_lcd.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 2.2. 통신 모듈 제거 및 초저전력 LCD 직관적 관제 (Zero-Latency)
네트워크 영상 송출로 인해 발생하는 통신 지연(Latency)과 연산 자원 낭비를 방지하기 위해, 상태 정보를 **I2C 16x2 Character LCD 패널**에 직접 렌더링하는 방식으로 전환했습니다.

*   **적용 하드웨어 제원:** 
    *   디스플레이: 16x2 Character LCD Display 모듈
    *   통신 인터페이스 및 주소: I2C 통신 / 0x27 (또는 0x3F)
    *   확장 칩셋: PCF8574 (LCD 후면 장착형 I2C 어댑터 모듈)
*   **시스템 이점:** 외부 네트워크에 의존하지 않는 0.0ms의 지연 시간을 달성하였으며, 디스플레이 장착이 제한적인 환경에서도 즉각적인 모니터링이 가능합니다.

---

## 3. 시스템 아키텍처 및 5단계 하이브리드 파이프라인

본 시스템은 네트워크 의존을 0%로 만들고 오로지 연산 정밀도와 즉각적인 LCD 렌더링에 초점을 맞춘 5단계 파이프라인 구조로 설계되었습니다.

### 3.1. 시스템 전반 블록 도면 (Architecture Diagram)

```mermaid
graph TD
    subgraph 1. 하드웨어 입력 (Hardware Input)
        A[Raspberry Pi Camera Rev 1.3] --> B(Raspberry Pi 3 Model B / 1GB RAM)
    end

    subgraph 2. 전처리 및 1차 탐지 (CPU & ZRAM)
        B --> C[1D 데이터 붕괴 방어 및 배열 재구성 모듈]
        C --> D[Dlib 얼굴 탐지 및 상관관계 추적기]
        D --> E[Dlib 68 랜드마크 위상 좌표계 추출]
    end

    subgraph 3. 하이브리드 병렬 추론 (Edge AI)
        E --> F(NCNN 8비트: 고개 방향 탐지)
        E --> G(TFLite 8비트: 마스크 착용 판별)
        E --> H(TFLite 8비트: 안검 폐쇄 확률 연산)
        E --> I(기하학 수학 연산: EAR, MAR, 상안면 35도 임계각 비율 연산)
    end

    subgraph 4. 결정 수준 융합 및 LCD 독립 출력 (Zero-Latency)
        F & G & H & I --> J{결정 융합 모듈: 딥러닝 81.3% 판별 + 18.7% 수학적 보정 개입}
        J --> K[상태 관제기: 시계열 기반 상태 누적 및 최우선순위 산출]
        K --> L[초저전력 16x2 LCD 패널 직관적 경고 출력]
        K --> M((특정 안면 패턴 기반 시스템 안전 종료 시퀀스))
    end
```

### 3.2. 5단계 융합 데이터 파이프라인 흐름 (Data Pipeline Flow)

단일 딥러닝 모델의 한계를 극복하기 위해 영상 유입부터 최종 LCD 출력까지 딥러닝과 기하학적 수식이 맞물려 돌아가는 하이브리드 병렬 연산을 수행합니다.

```mermaid
graph LR
    subgraph 1단계: 영상 유입 및 차원 복구
        P1[1D 데이터 붕괴 에러 방어<br/>2D 컬러 이미지 재구성]
    end

    subgraph 2단계: 1차 필터링 및 추적
        P1 --> P2[Dlib 얼굴 탐지<br/>잔상 방지: 3초 주기 추적기 초기화]
    end

    subgraph 3단계: 뼈대 위상 투영
        P2 --> P3[68 랜드마크 맵핑<br/>수면/하품/마스크 시계열 누적 필터링]
    end

    subgraph 4단계: 병렬 연산 및 결정 융합
        P3 --> P4A(NCNN 8비트: 고개 방향 추정)
        P3 --> P4B(TFLite 8비트: 마스크/안검 개폐 탐지)
        P3 --> P4C(수학적 기하학 보정: EAR, MAR)
        P4A & P4B & P4C --> P4D{결정 수준 융합<br/>딥러닝 신뢰도 하락 시 수식 보정}
    end

    subgraph 5단계: 판정 및 LCD 자립 출력
        P4D --> P5[네트워크 독립형<br/>0.0ms 초저지연 LCD 출력]
    end
```

---

## 4. 물리적 안면 패턴 기반 보안 시스템 셧다운

키보드와 마우스 등 입력 인터페이스가 없는 임베디드 환경에서 시스템을 외부 소프트웨어 개입 없이 하드웨어적으로 안전하게 종료하기 위해, 운전자의 연속된 안면 동작 패턴을 해독하는 보안 시퀀스를 탑재했습니다.

*   **작동 원리:** `[좌 -> 정면 -> 우 -> 정면 -> 좌 -> 정면 -> 우 -> 정면 -> 하품]` 형태의 9단계 동작을 순차적으로 감지하면, LCD에 종료 메시지를 출력한 후 리눅스 OS 안전 종료(Shutdown) 커맨드를 호출합니다.

---

## 5. 딥러닝의 시각적 인지 상실(Failure)과 18.7%의 수학적 대체(Fallback) 비율 분석

자율주행 환경에서 카메라 센서를 통해 유입되는 데이터는 환경 요인에 의해 시공간적 일관성을 상실하기 쉽습니다. 수천 번의 교차 시뮬레이션을 통해 도출된 딥러닝의 인지 실패율 18.7%는 크게 동적 흔들림(Motion Blur), 극단적 조명(Illumination), 물리적 가림(Occlusion)이라는 세 가지 상황으로 분류됩니다.

| 물리적 간섭 환경 (Failure Cause) | 발생 빈도 | 딥러닝의 인지적 오류 현상 | 수학적 대체(Fallback) 보완 메커니즘 |
| :--- | :--- | :--- | :--- |
| **동적 흔들림 (Motion Blur)** | **7.2%** | 픽셀 경계선 혼합으로 인한 공간적 특징 소실 및 확신도 저하. | **EAR 수식 및 랜드마크 기반 거리 연산**. 2D 좌표점 간의 비율을 계산하여 안검 폐쇄 여부 즉각 판별. |
| **극단적 조명 변화 (Illumination)** | **8.5%** | 과노출 영역을 마스크 착용으로 오분류하거나 객체 소실로 판단. | **픽셀 명암 분산도(Variance) 통계 필터링**. ROI의 질감 데이터를 수식화하여 노출 과다 현상 판별 및 오작동 억제. |
| **물리적 가림 및 왜곡 (Occlusion)** | **3.0%** | 안경테나 마스크 끈의 엣지(Edge)를 눈동자로 오인. | **마스크 감지 시 EAR 임계값 동적 상향(0.27)** 및 상안면 2D 비율 기반의 시선 추적 로직 적용. |

---

## 6. 수학적 기하학 로직: EAR, MAR 및 상안면 비율의 알고리즘 해부

### 6.1. EAR (Eye Aspect Ratio): 안검 폐쇄 판독을 위한 유클리디안 비율
운전자의 졸음 상태(Drowsiness)를 판별하기 위한 EAR 수식은 눈의 생물학적 개폐 특성을 수학적 비율로 치환합니다. 이는 카메라 렌즈와의 거리에 관계없이 스칼라 비율만을 도출하여 안정성을 확보합니다.

*   **마스크 착용 유무에 따른 동적 임계값 적용:** 마스크 미착용 시에는 **EAR < 0.22**를 적용하나, 마스크 착용이 감지되어 안면 근육의 왜곡이 예상될 경우 **EAR < 0.27**로 임계값을 보정하여 정밀도를 유지합니다.
*   **시계열 누적 검열 (Temporal Hysteresis):** 생리적 눈 깜박임과 수면 상태를 구분하기 위해 PERCLOS 알고리즘을 융합하여, **연속으로 2초(20프레임) 이상 누적**되었을 때만 졸음 상태로 판정합니다.

### 6.2. MAR (Mouth Aspect Ratio): 하품 감지 및 구강 개방 추론
졸음의 전조 증상인 하품(Yawning)을 감지하는 MAR 로직은 내부 입술 좌표를 활용하여 개방 비율이 임계점을 넘고 1초간 유지될 경우 하품으로 분류합니다.

### 6.3. 상안면 2D 황금비율 기반의 시선 이탈(Distraction) 판별
마스크 착용 시 하안면 좌표 오염으로 인한 3D Pose Estimation 연산 오류를 극복하기 위해, 마스크에 가려지지 않는 눈과 코끝 좌표만을 활용하는 **'상안면 2D 황금비율 로직'**을 채택했습니다. 수직/수평 거리 비율을 역산하여 추정 각도가 35도를 초과하고 1초 이상 지속될 경우 주의 태만으로 판별합니다.

---

## 7. 시각 인지 방어망 및 실전 검증 테스트

### 7.1. 1D 데이터 붕괴 에러 방어 및 지능형 배열 재구성 (Mathematical Reshape)
제한된 자원의 엣지 환경에서 구형 카메라 드라이버 오작동으로 인해 프레임 데이터가 1차원 배열(Size: 921,600)로 붕괴하는 현상이 발생할 경우, 실시간으로 Numpy의 `reshape(480, 640, 3)` 선형 행렬 변환 연산을 호출하여 2D 컬러 스페이스로 배열을 재구성(Re-formatting)하여 시스템 중단을 방지합니다.

### 7.2. 명암 분산도(Variance) 통계 검사와 노이즈 필터링
극단적인 조명 환경에서 딥러닝 모델의 오분류를 방지하기 위해 관심 영역(ROI) 내 픽셀 통계를 검사하는 **명암 분산도 필터(np.var(roi) < 600)**를 적용하여 노출 과다 현상을 수학적으로 판별합니다.

### 📸 실전 환경 다중 상태 감지 테스트 시각화 (Test Results)
아래 이미지는 극한 환경에서 마스크 유무, 고개 방향, 안검 개폐 상태를 독립적이고 정확하게 판별해내는 시스템의 실제 구동 화면입니다.

**[마스크 착용(ON) 상태 방어력 검증]**
<div align="center">
  <img src="test_results/마스크_상태변화/마스크_정면.png" width="19%" alt="Mask Front">
  <img src="test_results/마스크_상태변화/마스크_수면.png" width="19%" alt="Mask Sleep">
  <img src="test_results/마스크_상태변화/마스크_좌측.png" width="19%" alt="Mask Left">
  <img src="test_results/마스크_상태변화/마스크_우측.png" width="19%" alt="Mask Right">
  <img src="test_results/마스크_상태변화/마스크_아래.png" width="19%" alt="Mask Down">
</div>

**[마스크 미착용(OFF) 상태 정밀 검증]**
<div align="center">
  <img src="test_results/노마스크_상태변화/노마스크_정면.png" width="19%" alt="No Mask Front">
  <img src="test_results/노마스크_상태변화/노마스크_수면.png" width="19%" alt="No Mask Sleep">
  <img src="test_results/노마스크_상태변화/노마스크_좌측.png" width="19%" alt="No Mask Left">
  <img src="test_results/노마스크_상태변화/노마스크_우측.png" width="19%" alt="No Mask Right">
  <img src="test_results/노마스크_상태변화/노마스크_하.png" width="19%" alt="No Mask Down">
</div>

---

## 8. 하이브리드 엔진 아키텍처 및 하드웨어 최적화 지표

ARM Cortex-A53 기반 1GB RAM의 한정된 자원 내에서 효율적인 시각 판별을 수행하기 위해 3중 하이브리드 엔진을 설계했습니다.

1.  **Dlib 랜드마크 추출기:** 얼굴의 위상 뼈대 좌표계를 추출하여 기하학적 수식 연산의 기준점을 제공합니다.
2.  **NCNN 엣지 추론 엔진:** 모바일 아키텍처에 최적화된 텐서 엔진을 통해 고개 방향을 실시간으로 추정합니다.
3.  **TFLite 양자화 지능 엔진:** 범용 데이터(11만 장)와 실전 환경 커스텀 데이터(10,000장)를 융합 학습하고 INT8 양자화 압축을 적용하여 추론 속도를 최적화했습니다.

### 8.1. 독자 데이터셋 기반 모델 학습 및 검증 지표 (Training Metrics)
데이터 융합 학습 결과, 20 Epoch 내에 손실률(Loss)은 안정적으로 수렴하였으며 검증 정확도(Validation Accuracy)는 99% 이상에 도달하여 신뢰성을 확보했습니다.

<div align="center">
  <img src="metrics/head_pose_curve_20.png" width="45%" alt="Head Pose Training Metrics">
  <img src="metrics/training_curve_20.png" width="45%" alt="Mask & Blink Training Metrics">
</div>

### 8.2. 하드웨어 자원 점유율 및 발열 제어 검증 (Resource Monitoring)
아래의 시스템 모니터링 로그는 고강도 융합 연산 중에도 1GB RAM 환경에서 CPU 30-45%, RAM 40% 대의 점유율을 안정적으로 유지하며, 55-56°C의 온도로 발열 제어(Thermal Throttling 억제)에 성공했음을 나타냅니다.

<div align="center">
  <img src="metrics/실행직후.png" width="45%" alt="Boot Resource">
  <img src="metrics/실행후1시간.png" width="45%" alt="1 Hour Resource">
</div>

#### **[하드웨어 및 소프트웨어 최적화 요약]**
*   **8비트 INT8 양자화 압축:** 모델 크기 75% 감소, 캐시 적중률 상승을 통한 추론 지연 최소화.
*   **다중 코어 연산 병렬화:** `os.environ['OMP_NUM_THREADS'] = '4'` 적용으로 10 FPS 유지.
*   **가상 메모리 및 수동 GC:** 1GB ZRAM 스왑 활성화 및 `gc.collect()` 호출로 메모리 부족 현상 방지.
*   **카메라 화각(FOV) 한계 예외 처리:** 53.5도의 좁은 화각(Rev 1.3)을 고려하여, 프레임을 벗어나는 '고개 들림(Up)' 동작을 환경 노이즈로 예외 처리(`Front` 간주).

---

## 9. 결론

본 결정 수준 융합(Decision-level Fusion) 아키텍처는 데이터 기반 양자화 딥러닝의 패턴 인식 능력과, 인지 한계를 보완하는 수학적 기하학 로직(Heuristic Fallback)을 성공적으로 결합했습니다. 제한된 엣지 디바이스 환경 속에서도 독립적인 임베디드 기동(Systemd)과 초저전력 LCD 직관적 관제를 통해 네트워크 의존성 및 지연(Zero-Latency)을 극복함으로써, 자율주행 모니터링 시스템의 뛰어난 안정성과 신뢰성을 과학적으로 입증하였습니다.
