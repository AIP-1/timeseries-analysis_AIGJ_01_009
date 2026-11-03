"""
과제 4 연습용: 정상 데이터(synthetic/)의 마지막 1주에 고장 5종을 심어 synthetic_fault/ 를 만듦

  공식 데이터에는 고장이 이미 심어져 있으니 이 스크립트는 연습할 때만 씀.
  원본 1분 데이터(sensor.csv)를 직접 고쳐서, diagnose.py 의 정제를 그대로 거치게 함.

  고장 5종 (종류마다 3건, 모두 2026-09-28 ~ 10-04 시험 기간 안)
    튄 값        : 5분 한 칸만 값이 크게 튐 (CO2 +900ppm, 온도 +6.5℃, 습도 -12%)
    값 고정      : 1~4시간 동안 센서가 같은 숫자만 보냄 (고장 직전 값)
    CO2 영점     : 몇 시간 동안 CO2가 수백 ppm 높게 읽힘 (모양은 정상, 높이만 틀림)
    냉방 정지    : 수업 중 온도가 서서히 3~5℃ 오름 (수업 끝나고 1시간에 걸쳐 돌아옴)
    환기 정지    : 사용 중 CO2 상승폭이 평소의 1.8배 (끝나고 1시간에 걸쳐 돌아옴)

  결과
    synthetic_fault/sensor.csv   고장을 심은 센서 데이터 (schedule.csv, truth.csv 는 그대로 복사)
    synthetic_fault/faults.csv   정답: 고장 번호, 종류, 공간, 센서 항목, 시작, 끝  ← 채점에만 씀

사용:  python3 inject_faults.py                  # synthetic/ → synthetic_fault/
"""
import argparse
import csv
import os
import shutil
from collections import defaultdict
from datetime import datetime, timedelta

# (종류, 공간, 항목, 시작, 끝, 크기)
#   튄 값은 시작 시각의 5분 한 칸만 바꿈 (끝 = 시작 + 5분)
#   냉방 정지·환기 정지의 '끝'은 수업·회의가 끝나는 시각. 정답 구간은 회복 1시간까지 포함
FAULTS = [
    ("튄 값",     "강의실2", "co2",         "2026-09-28T15:20", None,               900),
    ("튄 값",     "라운지",  "temperature", "2026-09-30T03:10", None,               6.5),
    ("튄 값",     "회의실",  "humidity",    "2026-10-03T11:40", None,               -12),
    ("값 고정",   "강의실1", "co2",         "2026-09-28T13:10", "2026-09-28T15:40", None),
    ("값 고정",   "회의실",  "temperature", "2026-09-29T12:30", "2026-09-29T16:30", None),
    ("값 고정",   "라운지",  "co2",         "2026-10-02T11:00", "2026-10-02T12:00", None),
    ("CO2 영점",  "강의실2", "co2",         "2026-09-29T06:00", "2026-09-29T10:00", 350),
    ("CO2 영점",  "회의실",  "co2",         "2026-10-01T13:00", "2026-10-01T16:00", 300),
    ("CO2 영점",  "라운지",  "co2",         "2026-10-03T09:00", "2026-10-03T14:00", 250),
    ("냉방 정지", "강의실1", "temperature", "2026-09-30T14:00", "2026-09-30T16:50", 4.0),
    ("냉방 정지", "강의실2", "temperature", "2026-10-01T09:00", "2026-10-01T12:50", 5.0),
    ("냉방 정지", "강의실1", "temperature", "2026-10-02T09:00", "2026-10-02T11:50", 3.0),
    ("환기 정지", "강의실2", "co2",         "2026-09-28T10:00", "2026-09-28T12:50", 1.8),
    ("환기 정지", "강의실1", "co2",         "2026-10-01T13:00", "2026-10-01T17:50", 1.8),
    ("환기 정지", "회의실",  "co2",         "2026-10-02T10:00", "2026-10-02T11:30", 1.8),
]
RECOVER = 60          # 냉방·환기 정지 뒤 원래대로 돌아오는 시간(분)
DIGITS = {"co2": 1, "temperature": 2, "humidity": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="synthetic")
    ap.add_argument("--out", default="synthetic_fault")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.dir, "sensor.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # (공간, 항목) → {1분 시각: [행 번호, ...]}  (중복 행이 있어서 목록으로 둠)
    where = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        where[(r["space"], r["item"])][datetime.fromisoformat(r["time"])].append(i)

    def get(sp, item, t):
        idx = where[(sp, item)].get(t)
        return float(rows[idx[0]]["value"]) if idx else None

    def put(sp, item, t, v):
        for i in where[(sp, item)].get(t, []):     # 결측(행 없음)이면 아무것도 안 함
            rows[i]["value"] = f"{v:.{DIGITS[item]}f}"

    def minutes(st, en):
        t = st
        while t < en:
            yield t
            t += timedelta(minutes=1)

    answer = []
    for k, (kind, sp, item, st, en, size) in enumerate(FAULTS, 1):
        st = datetime.fromisoformat(st)
        en = st + timedelta(minutes=5) if en is None else datetime.fromisoformat(en)
        ans_end = en                                       # 정답 구간의 끝

        if kind == "튄 값":
            for t in minutes(st, en):                      # 5분 칸 하나를 통째로
                v = get(sp, item, t)
                if v is not None:
                    put(sp, item, t, v + size)

        elif kind == "값 고정":
            last = next(get(sp, item, st - timedelta(minutes=m)) for m in range(1, 30)
                        if get(sp, item, st - timedelta(minutes=m)) is not None)
            for t in minutes(st, en):
                put(sp, item, t, last)

        elif kind == "CO2 영점":
            for t in minutes(st, en):
                v = get(sp, item, t)
                if v is not None:
                    put(sp, item, t, v + size)

        elif kind == "냉방 정지":
            dur = (en - st).total_seconds() / 60
            ans_end = en + timedelta(minutes=RECOVER)
            for t in minutes(st, ans_end):
                m = (t - st).total_seconds() / 60
                # 수업 중엔 0 → size 로 서서히 오르고, 끝난 뒤 1시간에 걸쳐 0으로
                add = size * m / dur if m < dur else size * (1 - (m - dur) / RECOVER)
                v = get(sp, item, t)
                if v is not None:
                    put(sp, item, t, v + add)

        elif kind == "환기 정지":
            # 기준 = 시작 전 10분 평균. 기준보다 오른 만큼(상승폭)을 1.8배로
            pre = [get(sp, item, st - timedelta(minutes=m)) for m in range(1, 11)]
            base = sum(v for v in pre if v is not None) / sum(v is not None for v in pre)
            dur = (en - st).total_seconds() / 60
            ans_end = en + timedelta(minutes=RECOVER)
            for t in minutes(st, ans_end):
                m = (t - st).total_seconds() / 60
                factor = size if m < dur else 1 + (size - 1) * (1 - (m - dur) / RECOVER)
                v = get(sp, item, t)
                if v is not None and v > base:
                    put(sp, item, t, base + (v - base) * factor)

        answer.append([k, kind, sp, item, st.isoformat(), ans_end.isoformat()])
        print(f"  {k:2}. {kind:7} {sp:5} {item:12} {st:%m-%d %H:%M} ~ {ans_end:%m-%d %H:%M}")

    with open(os.path.join(a.out, "sensor.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time", "space", "item", "value"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "faults.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fault_id", "type", "space", "item", "start", "end"])
        w.writerows(answer)
    for name in ("schedule.csv", "truth.csv"):
        if os.path.exists(os.path.join(a.dir, name)):
            shutil.copy(os.path.join(a.dir, name), os.path.join(a.out, name))
    print(f"\n  저장: {a.out}/sensor.csv, faults.csv (+ schedule.csv, truth.csv 복사)")


if __name__ == "__main__":
    main()
