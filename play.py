#!/usr/bin/env python3
"""
ZeroForge - 象棋对弈启动器
支持动态加载检查点、UCI 引擎对接
"""

import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="ZeroForge 象棋对弈启动器")
    parser.add_argument("--share", action="store_true", help="是否开启 Gradio 外网分享")
    parser.add_argument("--port", type=int, default=7860, help="Web 服务端口")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU（不使用 GPU）")
    
    args = parser.parse_args()
    
    # 在导入 JAX 之前设置 CPU 模式
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        print("[设备] 强制使用 CPU 模式")
    
    print("="*50)
    print("ZeroForge Web GUI 启动中...")
    print(f"本地地址: http://127.0.0.1:{args.port}")
    print("功能说明:")
    print("1. 红黑方均可在界面选择: Human / ZeroForge AI / UCI Engine")
    print("2. AI 检查点可在 'AI/UCI 配置' 标签页动态加载，无需重启")
    print("3. 支持导入/导出 FEN 局面")
    print("4. 界面已适配移动端自适应布局")
    if args.cpu:
        print("5. 当前运行在 CPU 模式")
    print("="*50)
    
    # 延迟导入，确保 JAX_PLATFORMS 环境变量生效
    from gui.web_gui import create_ui
    demo = create_ui()
    demo.launch(
        share=args.share,
        server_name="127.0.0.1",
        server_port=args.port
    )

if __name__ == "__main__":
    main()
