@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: 设置配置
:: ============================================================
set "SOURCE_ROOT=chathistory"
set "TARGET_DIR=result"
set "OUTPUT_FILE=bat_result.txt"

:: 检查是否存在 chathistory 目录
if not exist "%SOURCE_ROOT%" (
    echo [错误] 找不到目录: %SOURCE_ROOT%
    pause
    exit /b 1
)

:: 检查是否存在 evaluate.py
if not exist "evaluate.py" (
    echo [错误] 找不到脚本: evaluate.py
    pause
    exit /b 1
)

:: 如果输出文件不存在，创建一个空的；如果存在，追加分隔符
if not exist "%OUTPUT_FILE%" (
    type nul > "%OUTPUT_FILE%"
) else (
    echo. >> "%OUTPUT_FILE%"
    echo ======================================================== >> "%OUTPUT_FILE%"
    echo 新的执行开始: %date% %time% >> "%OUTPUT_FILE%"
    echo ======================================================== >> "%OUTPUT_FILE%"
)

:: ============================================================
:: 主循环：遍历 chathistory 下包含 "critic" 的文件夹
:: ============================================================
echo 正在搜索包含 "critic" 的文件夹...

for /d %%D in ("%SOURCE_ROOT%\*critic*") do (
    set "FOLDER_NAME=%%~nxD"
    set "FULL_SOURCE_PATH=%%D\traj\result"
    set "CURRENT_CRITIC_DIR=%%D"
    
    echo.
    echo --------------------------------------------------------
    echo 处理文件夹: !FOLDER_NAME!
    echo 源路径: !FULL_SOURCE_PATH!

    :: 检查源路径下的 result 目录是否存在
    if exist "!FULL_SOURCE_PATH!" (
        
        :: 1. 复制/替换文件
        :: /E: 复制目录和子目录 /Y: 禁止提示覆盖 /I: 如果目标不存在则假定为目录
        echo 正在将文件复制到当前目录的 %TARGET_DIR% ...
        xcopy "!FULL_SOURCE_PATH!\*" "%TARGET_DIR%\" /E /Y /I > nul

        :: 2. 记录文件夹名称到结果文件
        echo. >> "%OUTPUT_FILE%"
        echo !FOLDER_NAME! >> "%OUTPUT_FILE%"

        :: 3. 执行 Python 脚本并捕获最后 9 行输出
        echo 正在执行 evaluate.py ...
        
        :: 使用临时文件捕获所有输出
        python evaluate.py --dataset 2019 --mode 1 > temp_output.log 2>&1
        
        :: 使用 PowerShell 提取最后 9 行并追加到 bat_result.txt
        powershell -Command "Get-Content temp_output.log -Tail 10 | Out-File -Append -FilePath '%OUTPUT_FILE%' -Encoding UTF8"
        
        :: 也可以在控制台显示一下以便监控进度
        powershell -Command "Get-Content temp_output.log -Tail 10"

        :: 清理临时文件
        del temp_output.log

        :: 4. [新增] 将当前目录下的所有 png 图片剪切到对应的 critic 文件夹
        echo 正在移动 PNG 图片到: !CURRENT_CRITIC_DIR! ...
        
        :: 检查当前目录下是否有 png 文件，避免报错
        if exist "*.png" (
            move "*.png" "!CURRENT_CRITIC_DIR!\" > nul
            echo [成功] 图片已移动。
        ) else (
            echo [提示] 当前目录下没有找到 PNG 图片。
        )

    ) else (
        echo [警告] 跳过: 在 "!FOLDER_NAME!" 中未找到 traj\result 路径
    )
)

echo.
echo ========================================================
echo 所有任务已完成。结果已保存至 %OUTPUT_FILE%
echo ========================================================
pause