# -*- coding: utf-8 -*-
"""
test_app.py —— 用 Streamlit AppTest 验证应用各页面可正常渲染
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py")

PAGES = ["📊 数据总览", "🎯 模型训练", "✂️ 剪枝对比", "📜 规则解读", "🔮 单客户预测"]

print("=" * 60)
print("Streamlit 应用页面测试")
print("=" * 60)

ok_count = 0
for page in PAGES:
    try:
        at = AppTest.from_file(APP, default_timeout=180)
        at.run()
        if len(at.exception) > 0:
            print(f"  [FAIL] 初始加载异常：{at.exception[0].value}")
            break

        # 切换到目标页面
        at.sidebar.radio[0].set_value(page).run()
        if len(at.exception) > 0:
            print(f"  [FAIL] {page}：{at.exception[0].value}")
            continue

        # 统计渲染元素
        n_md = len(at.markdown)
        n_metrics = len(at.metric) if hasattr(at, "metric") else 0
        n_tabs = len(at.tabs) if hasattr(at, "tabs") else 0
        print(f"  [OK]   {page:14s} markdown={n_md:3d} metric={n_metrics:2d} tabs={n_tabs}")
        ok_count += 1
    except Exception as e:
        print(f"  [ERROR] {page}: {type(e).__name__}: {e}")

print("=" * 60)
print(f"通过 {ok_count}/{len(PAGES)} 个页面")
print("=" * 60)