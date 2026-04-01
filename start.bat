@echo off
chcp 65001 >nul
echo ============================================================
echo   基于知识图谱和大模型的 Java 课程问答系统 - 一键启动
echo ============================================================

:: 设置 JAVA_HOME
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot

:: 启动 Neo4j
echo [1/2] 正在启动 Neo4j 图数据库...
start "" "%~dp0neo4j-community-5.26.0\bin\neo4j.bat" console

:: 等待 Neo4j 启动
echo 等待 Neo4j 启动 (15秒)...
timeout /t 15 /nobreak >nul

:: 启动 Flask
echo [2/2] 正在启动 Flask 后端...
cd /d "%~dp0"
start "" python app.py

:: 等待 Flask 启动
timeout /t 5 /nobreak >nul

echo.
echo ============================================================
echo   系统已启动！
echo   浏览器打开: http://localhost:5000
echo   Neo4j 控制台: http://localhost:7474
echo ============================================================
echo.
echo 按任意键打开浏览器...
pause >nul
start http://localhost:5000
