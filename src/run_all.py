# -*- coding: utf-8 -*-
"""
run_all.py —— 一键运行全部实验
================================
用法：python run_all.py
"""
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")


def run(name, module):
    print("\n" + "#" * 60)
    print(f"#  运行 {name}")
    print("#" * 60)
    t0 = time.time()
    try:
        if module == "exp01":
            from exp01_exploration import iris_warmup, telco_exploration
            iris_warmup()
            telco_exploration()
        elif module == "exp02":
            from exp02_baseline import main as m
            m()
        elif module == "exp03":
            from exp03_experiments import main as m
            m()
        elif module == "exp04":
            from exp04_challenges import main as m
            m()
        print(f"\n[OK] {name} 耗时 {time.time()-t0:.1f}s")
        return True
    except Exception as e:
        import traceback
        print(f"\n[FAIL] {name}: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    t_all = time.time()
    results = [
        ("数据探索", "exp01"),
        ("数据预处理与基准模型", "exp02"),
        ("4 组必做对比实验", "exp03"),
        ("3 个挑战任务", "exp04"),
    ]
    status = [(n, run(n, m)) for n, m in results]

    print("\n" + "=" * 60)
    print("运行汇总")
    print("=" * 60)
    for n, ok in status:
        print(f"  {'✓' if ok else '✗'} {n}")
    print(f"\n总耗时：{time.time()-t_all:.1f}s")
    print(f"结果目录：{os.path.join(os.path.dirname(BASE_DIR), 'results')}")
    print(f"图表目录：{os.path.join(os.path.dirname(BASE_DIR), 'images')}")
    print("=" * 60)