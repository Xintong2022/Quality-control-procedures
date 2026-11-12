@echo off
cd /d "%~dp0"
echo [1/2] 安装依赖（如已装可跳过）...
python -m pip install -r requirements.txt
echo [2/2] 启动服务（新窗口）并打开浏览器...
start "TravelQA" python server.py
timeout /t 4 >nul
start http://localhost:8501
echo.
echo 服务已在后台窗口运行。关闭该窗口即可停止服务。
echo 浏览器若未自动打开，请手动访问 http://localhost:8501
pause
