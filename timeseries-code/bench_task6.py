"""
과제 6-4 비용 재기: 모델 1회 추론 시간과 메모리 사용량 (보드에서 그대로 실행)

  모델마다 새 프로세스를 띄워서 따로 잼 (앞 모델이 쓴 메모리가 섞이지 않게)
    - 불러오기 시간, 파라미터 수, 파일 크기
    - 1회 추론 시간: 공간 1곳(배치 1) / 공간 4곳 한 번에(배치 4), 30번 재서 중앙값
    - 메모리: 프로세스 최대 사용량(RSS), torch 만 불러 둔 상태와의 차이
      GPU(cuda)로 돌리면 GPU 메모리 최대 사용량도 같이 기록
      ※ Jetson 은 CPU·GPU 가 메모리를 나눠 쓰므로 두 값을 더한 것이 실제 부담에 가까움

사용:  python3 bench_task6.py                         # CPU, 모든 모델
       python3 bench_task6.py --device cuda           # Jetson GPU
       python3 bench_task6.py --models chronos-bolt-small,gru_sched --context 2016
결과: diagnosis/bench_<장치>.csv
같은 폴더에 train_task3.py, diagnose.py, task6_models/, diagnosis/gru_*.pt 가 있어야 함
"""
import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ALL = ["gru_sched", "gru_nosched", "chronos-bolt-tiny", "chronos-bolt-small", "chronos-2", "chronos-2+cov"]
# chronos-2+cov = Chronos-2 에 온도·습도·시각·주말·일정(앞으로 60분 포함)까지 함께 넣은 경우


def rss_mb():
    """지금까지 이 프로세스가 쓴 최대 메모리(MB)"""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if platform.system() == "Darwin" else r / 1024   # 맥은 바이트, 리눅스는 KB


def now_rss_mb():
    """지금 쓰고 있는 메모리(MB). 리눅스(/proc)에서만, 아니면 최대값으로 대신"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return rss_mb()


def child(name, device, context, runs, threads):
    """새 프로세스 안에서 모델 하나를 잼 → JSON 한 줄 출력"""
    import numpy as np
    import torch
    if threads:
        torch.set_num_threads(threads)
    sys.path.insert(0, HERE)
    base = now_rss_mb()
    t_load = time.perf_counter()
    if name.startswith("gru"):
        import train_task3 as T
        sched = name == "gru_sched"
        n_feat = 9 + (1 if sched else 0) + 4          # train_task3.build_features 와 같은 특성 수
        model = T.GRUForecaster(n_feat, T.FUTURE if sched else 0, 64)
        path = os.path.join(HERE, "diagnosis", f"{name}.pt")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.to(device).eval()
        n_param = sum(p.numel() for p in model.parameters())

        def run(b):
            x = torch.randn(b, T.WINDOW, n_feat, device=device)
            fut = torch.zeros(b, T.FUTURE if sched else 0, device=device)
            with torch.no_grad():
                model(x, fut)
    else:
        from chronos import BaseChronosPipeline
        use_cov = name.endswith("+cov")
        path = os.path.join(HERE, "task6_models", name.replace("+cov", ""))
        pipe = BaseChronosPipeline.from_pretrained(path, device_map=device, torch_dtype=torch.float32)
        n_param = sum(p.numel() for p in pipe.model.parameters())
        is2 = type(pipe).__name__ == "Chronos2Pipeline"

        def run(b):
            ctx = 400 + 50 * np.random.rand(b, context).astype(np.float32)
            if use_cov:
                past = lambda: np.random.rand(context).astype(np.float32)
                fut = lambda: np.random.rand(12).astype(np.float32)
                x = [{"target": c,
                      "past_covariates": {k: past() for k in ("temperature", "humidity", "tod_sin", "tod_cos",
                                                              "weekend", "schedule")},
                      "future_covariates": {k: fut() for k in ("tod_sin", "tod_cos", "weekend", "schedule")}}
                     for c in ctx]
            else:
                x = torch.from_numpy(ctx[:, None, :] if is2 else ctx)
            pipe.predict_quantiles(x, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])
    load_s = time.perf_counter() - t_load
    after_load = now_rss_mb()
    size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 1e6 \
        if os.path.isdir(path) else os.path.getsize(path) / 1e6
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out = {"model": name, "device": device, "context": context if not name.startswith("gru") else 24,
           "params_M": round(n_param / 1e6, 2), "file_MB": round(size_mb, 1), "load_s": round(load_s, 2)}
    for b in (1, 4):
        for _ in range(3):                 # 처음 몇 번은 준비 시간이 섞여서 버림
            run(b)
        ts = []
        for _ in range(runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            run(b)
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        out[f"ms_batch{b}"] = round(float(np.median(ts)), 2)
    out["rss_torch_only_MB"] = round(base, 0)
    out["rss_after_load_MB"] = round(after_load, 0)
    out["rss_peak_MB"] = round(rss_mb(), 0)
    out["rss_peak_minus_torch_MB"] = round(rss_mb() - base, 0)
    out["gpu_peak_MB"] = round(torch.cuda.max_memory_allocated() / 1e6, 0) if device == "cuda" else ""
    print("RESULT " + json.dumps(out, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--context", type=int, default=2016, help="사전학습 모델에 주는 과거 칸 수 (5분 칸, 2016=7일)")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0, help="CPU 스레드 수 (0이면 기본값)")
    ap.add_argument("--child", default="")
    a = ap.parse_args()
    if a.child:
        child(a.child, a.device, a.context, a.runs, a.threads)
        return

    rows = []
    for name in a.models.split(","):
        p = subprocess.run([sys.executable, __file__, "--child", name, "--device", a.device,
                            "--context", str(a.context), "--runs", str(a.runs), "--threads", str(a.threads)],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(f"  {name}: 실패\n{p.stderr[-800:]}")
            continue
        r = json.loads(line[7:])
        rows.append(r)
        print(f"  {name:20} 파라미터 {r['params_M']:6.2f}M  파일 {r['file_MB']:6.1f}MB  "
              f"1회(공간1) {r['ms_batch1']:8.1f}ms  1회(공간4) {r['ms_batch4']:8.1f}ms  "
              f"메모리 최대 {r['rss_peak_MB']:6.0f}MB (torch만 {r['rss_torch_only_MB']:.0f}MB)"
              + (f"  GPU {r['gpu_peak_MB']}MB" if r["gpu_peak_MB"] != "" else ""))
    os.makedirs(os.path.join(HERE, "diagnosis"), exist_ok=True)
    out = os.path.join(HERE, "diagnosis", f"bench_{a.device}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  저장: {out}   (기계: {platform.machine()} / {platform.platform()})")


if __name__ == "__main__":
    main()
