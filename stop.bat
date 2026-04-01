@echo off
chcp 65001 >nul
echo 正在停止服务...

:: 停止 Flask
taskkill /f /im python.exe 2>nul

:: 停止 Neo4j
taskkill /f /im java.exe 2>nul

echo 所有服务已停止。
pause
