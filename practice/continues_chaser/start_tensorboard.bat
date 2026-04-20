@echo off
:: 1. 初始化 Conda 路径 (如果 conda 不在系统变量中，需指定路径)
:: 通常可以使用 'call activate'，但在脚本中 'call' 是必须的
call conda activate myml

:: 2. 启动 TensorBoard
echo Starting TensorBoard...
tensorboard --logdir runs

:: 3. 保持窗口开启（以防启动失败看不到报错）
pause