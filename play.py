#!/usr/bin/env python3
"""
ZeroForge - 象棋对弈启动器
支持动态加载检查点、UCI 引擎 (Pikafish)
"""

import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="ZeroForge 象棋对弈启动器")
    parser.add_argument("--host", default="127.0.0.1", help="服务器地址 (默认: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Web 服务端口 (默认: 7860)")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU（不使用 GPU）")
    
    args = parser.parse_args()
    
    # 在导入 JAX 之前设置 CPU 模式
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        print("[设备] 强制使用 CPU 模式")
    
    # 延迟导入，确保 JAX_PLATFORMS 环境变量生效
    from gui.web_gui import run_web_gui
    run_web_gui(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
