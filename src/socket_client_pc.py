# 파일명: src/socket_client_pc.py
# [혁신 모빌리티 AI - 관리자 로컬 PC 전용 초저지연 영상 수신기]

import cv2
import socket
import struct
import numpy as np

def start_socket_client():
    # 1. 라즈베리파이(서버)와 연결하기 위한 클라이언트 소켓 생성
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 주의: 실제 구동 중인 라즈베리파이의 IP 주소로 변경해야 합니다.
    RPI_IP = '192.168.3.32' 
    PORT = 9999
    
    print(f"⚙️ 라즈베리파이({RPI_IP}:{PORT})로 다이렉트 소켓 연결을 시도합니다...")
    try:
        client_socket.connect((RPI_IP, PORT))
        print("✅ 연결 성공! 영상 수신을 시작합니다. (종료 시 'q' 키 입력)")
    except Exception as e:
        print(f"❌ 연결 실패! 라즈베리파이 서버가 구동 중인지 확인해 주세요. 에러: {e}")
        return

    data = b""
    # 4바이트 크기의 데이터(영상 크기 정보) 규격 계산
    payload_size = struct.calcsize(">L")

    try:
        while True:
            # 1. 4바이트 크기의 패킷(영상 크기 정보)이 모두 수신될 때까지 대기
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet: break
                data += packet
            
            if not data: break

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            
            # 수신된 패킷을 해독하여 현재 프레임의 실제 데이터 크기(Byte) 추출
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            # 2. 추출한 크기만큼 실제 영상 데이터를 수신 및 병합
            while len(data) < msg_size:
                data += client_socket.recv(4096)
            
            frame_data = data[:msg_size]
            data = data[msg_size:]

            # 3. 수신 완료된 JPEG 영상 데이터를 Numpy 배열로 해독하여 화면에 출력
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            
            # 모니터링 환경에 맞추어 640x480 해상도로 출력 크기 조정
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow("HQ Command Center Monitor", frame)

            # 'q' 키 입력 시 즉시 수신 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"🚨 수신 중 에러 발생: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("=== 시스템 수신 종료 ===")

if __name__ == '__main__':
    start_socket_client()
