과제 6  사전학습 시계열 모델(Chronos) 비교 — 실행 안내
====================================================

폴더 구성
  diagnose.py, baseline.py, train_task3.py   과제 2·3과 같은 정제·기간·평가 시점·GRU
  task6_compare.py                           정확도·10~90% 구간 적중률·상황별·과거 길이별 비교
  bench_task6.py                             1회 추론 시간·메모리 (보드에서 실행)
  synthetic/                                 연습 합성 데이터 (실제 키트 데이터로 바꿔 넣기)
  diagnosis/                                 연습 데이터로 돌린 결과 (다시 돌리면 덮어씀)
  task6_models/                              ← 여기에 모델 파일을 넣어야 함 (zip 에는 없음, 너무 커서)
      chronos-bolt-tiny/  config.json, model.safetensors   (35MB)
      chronos-bolt-small/ config.json, model.safetensors   (191MB)
      chronos-2/          config.json, model.safetensors   (478MB)
    맥의 ~/Documents/과제 3/task6_models 를 그대로 복사하면 됨.
    새로 받을 때:  https://huggingface.co/amazon/<모델이름>/resolve/main/config.json
                  https://huggingface.co/amazon/<모델이름>/resolve/main/model.safetensors

설치
  pip install torch chronos-forecasting        (맥·PC)
  Jetson: JetPack 에 맞는 torch 휠을 먼저 설치한 뒤  pip install chronos-forecasting --no-deps
          그리고 pip install transformers accelerate einops pandas  (chronos 가 쓰는 것)

실행 순서 (맥·PC, 결과 숫자)
  python3 baseline.py              # diagnosis/norm_stats.csv, eval_points.csv, baseline_results.csv
  python3 train_task3.py           # diagnosis/gru_sched.pt, gru_nosched.pt (과제 3 GRU)
  python3 task6_compare.py         # 전체 비교. CPU 4개로 약 1시간 반 (Chronos-2 가 느림)
  python3 task6_compare.py --quick --chronos chronos-bolt-small      # 빠르게 확인 (몇 분)
  결과: diagnosis/task6_results.csv (모든 평가 시점)
        diagnosis/task6_results_sub.csv (다른 변수를 넣은 Chronos-2 포함, 12개 중 1개 시점)
        diagnosis/task6_situation.csv, task6_context.csv, task6_preds.npz

보드(Jetson)에서 비용 재기
  python3 bench_task6.py --device cuda         # GPU
  python3 bench_task6.py --device cpu          # CPU 만
  python3 bench_task6.py --models chronos-bolt-small,gru_sched --context 2016
  결과: diagnosis/bench_<장치>.csv
  ※ Jetson 은 CPU·GPU 가 같은 메모리를 나눠 씀 → "메모리 최대(RSS) + GPU 최대" 가 실제 부담에 가까움
  ※ 재는 동안 다른 프로그램(특히 로컬 LLM)을 끄고 한 번, 켜고 한 번 재면 "함께 올렸을 때"를 볼 수 있음
     free -m 또는 tegrastats 로 전체 메모리 여유도 같이 적어 두기

주의
  모든 숫자는 연습 합성 데이터 기준. 제출 전 실제 키트 데이터(synthetic/ 자리에)로 다시 돌려 숫자를 바꿀 것.
  truth.csv(실제 인원)는 상황별 채점에만 쓰고 어떤 모델의 입력에도 넣지 않음.
