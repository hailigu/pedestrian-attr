@echo off
set num=%1
set src=%2
set dest=%3

for /l %%i in (1,1,%num%) do (
	setlocal enabledelayedexpansion
	
	rem ���㵱ǰĿ¼���ж��ٸ�ͼƬ->curnum
	set /a curnum=0
	for %%m in (%src%\*.jpg)do (
		set /a curnum+=1
	)
	@echo ��ǰĿ¼����!curnum!��ͼƬ
	
	rem ����һ��0~curnum-1֮��������index
	set /a index=!random!%%!curnum!
	@echo ���ѡȡ��!index!��ͼƬ
	
	rem �ƶ���index��ͼƬ��Ŀ��Ŀ¼
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
	@echo �ƶ���!index!��ͼƬ��Ŀ��Ŀ¼

:end
