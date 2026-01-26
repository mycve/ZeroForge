"""
ZeroForge Web GUI - 启动入口

新版 GUI 使用 Vue + FastAPI 架构，提供更流畅的用户体验。
- 前端: Vue 3 + Tailwind CSS (CDN)
- 后端: FastAPI + Uvicorn

特性:
- UCI 引擎 (Pikafish) 全局单例，自动启动
- 每次新局自动清空 UCI hash
- 支持 ZeroForge AI 和 UCI 引擎对弈
"""

import argparse


def run_web_gui(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    """
    启动 Web GUI 服务器
    
    Args:
        host: 服务器地址，默认 0.0.0.0 允许外部访问
        port: 服务器端口，默认 7860
        share: 是否创建公共链接（暂不支持，保留兼容性）
    """
    import uvicorn
    from gui.api import app
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║           ZeroForge 象棋对弈 - Web GUI                      ║
╠═══════════════════════════════════════════════════════════╣
║  访问地址: http://{host}:{port}                             
║  本机访问: http://127.0.0.1:{port}                          
║                                                           ║
║  按 Ctrl+C 停止服务器                                       ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="ZeroForge 象棋对弈 Web GUI")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口 (默认: 7860)")
    parser.add_argument("--share", action="store_true", help="创建公共链接 (暂不支持)")
    args = parser.parse_args()
    
    run_web_gui(host=args.host, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
