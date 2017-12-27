@echo off
set num=%1
set src=%2
set dest=%3

for /l %%i in (1,1,%num%) do (
	setlocal enabledelayedexpansion
	
	rem 计算当前目录下有多少个图片->curnum
	set /a curnum=0
	for %%m in (%src%\*.jpg)do (
		set /a curnum+=1
	)
	@echo 当前目录中有!curnum!张图片
	
	rem 生成一个0~curnum-1之间的随机数index
	set /a index=!random!%%!curnum!
	@echo 随机选取第!index!张图片
	
	rem 移动第index张图片到目标目录
	call :startmov

	endlocal
) 

goto end

:startmov
	set /a curnum=0
	for %%m in (%src%\*.jpg)do (
		if !curnum! equ !index! (
			echo %%m
			move "%%m" %dest%
			call :movok
		)
		set /a curnum+=1
	)
	goto end
	
	:movok
	@echo 移动第!index!张图片到目标目录

:end
