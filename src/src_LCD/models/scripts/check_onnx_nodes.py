# 파일명: check_onnx_nodes.py
# 실행 방법: (hailo_dfc_env) j2w@j2w-notebook:~/AI$ python3 check_onnx_nodes.py

import onnx

def inspect_onnx_nodes():
    print("="*60)
    print(" ONNX 모델 입출력 노드(Node) 형태(Shape) 검증 ")
    print("="*60)
    
    onnx_path = "drowsy_eyes.onnx"
    
    try:
        # 가중치(Weights) 데이터를 제외하고 모델의 그래프(Graph) 구조만 메모리에 로드하여 분석 속도 최적화.
        model = onnx.load(onnx_path, load_external_data=False)
        
        print("\n[입력 노드 (Inputs)]")
        for inp in model.graph.input:
            print(f" - 노드명: '{inp.name}' / 형태(Shape): {inp.type.tensor_type.shape}")
            
        print("\n[출력 노드 (Outputs)]")
        for out in model.graph.output:
            print(f" - 노드명: '{out.name}' / 형태(Shape): {out.type.tensor_type.shape}")
            
        print("\n" + "="*60)
        print("ONNX 모델 노드 분석 완료. 상기 입출력 명세 확인 요망.")
        print("="*60)
        
    except Exception as e:
        print(f"[오류] ONNX 모델 로드 중 예외 발생: {e}")

if __name__ == "__main__":
    inspect_onnx_nodes()
    
