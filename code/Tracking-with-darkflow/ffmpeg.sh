#!/bin/sh
if [ ! -n "$1" ] ;then
	echo "you have not input a video!"
        filename=output_test.h264
else
	filename=$1
fi

filesize=`ls -l $filename | awk '{ print $5 }'`
maxsize=$((10*1024*1024))

# wait for push
while [ $filesize -lt $maxsize ];do
	filesize=`ls -l $filename | awk '{ print $5 }'`
	echo $filesize
	sleep 1
done

# push
tmpfile=media"`date +%Y-%m-%d_%H:%M:%S`_"$filename
cp $filename $tmpfile
sleep 1
ffmpeg -re -i $tmpfile -vcodec libx264  -preset ultrafast -f flv rtmp://video-center-bj.alivecdn.com/app/stream?vhost=live.hailigu.com


