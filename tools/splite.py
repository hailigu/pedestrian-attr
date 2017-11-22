# coding=gb2312
import string
import os
import time
import re
import math
import sys
from optparse import OptionParser

print "Test by gongjia start..."
print (sys.path[0])
#print (sys.argv[0])
print (os.getcwd())
#print (os.path.abspath(__file__))
#print (os.path.realpath(__file__))

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", action="store_true", help="input x y for each file by user")
parser.add_option("-q", "--quality", dest="q", action="store", help="input xvid q arg", default="24")
parser.add_option("-v", "--vcodec", dest="vcodec", action="store", help="input video codec", default="x264")
parser.add_option("-n", "--noaudio", dest="an", action="store_true", help="no audio")
parser.add_option("-p", "--preset", dest="preset", action="store", help="", default="")
parser.add_option("-m", "--maxWidth", dest="maxWidth", action="store", help="input max width for output video",
                  default="")
parser.add_option("-f", "--fileType", dest="fileType", action="store", help="", default="mp4")
parser.add_option("-o", "--ogg", dest="ogg", action="store_true", help="user ogg instead of aac", default="")
parser.add_option("-3", "--mp3", dest="mp3", action="store_true", help="user mp3 instead of aac", default="")
parser.add_option("-1", "--pad", dest="pad", action="store_true", help="pad to 16:9", default="")
parser.add_option("-s", "--src", dest="srcD", action="store", help="source dir",
                  default=os.getcwd())
parser.add_option("-t", "--target", dest="targetD", action="store", help="target dir",
                  default=os.getcwd())
parser.add_option("-w", "--workdir", dest="workdir", action="store", help="work dir",
                  default=os.getcwd())
# 按照size切割
parser.add_option("-e", "--split", dest="split", action="store_true", help="split to multiple file with size")

# 按照时间切割
parser.add_option("-d", "--splitsize", dest="splitsize", action="store", help="split to multiple file with size",
                  default="3")  # Minutes

parser.add_option("-j", "--prefix", dest="prefix", action="store", help="target file name prefix", default="")

(options, args) = parser.parse_args()

if options.srcD == None or options.srcD[0:1] == '-':
    print 'srcD Err, quit'
    exit()
if options.targetD == None or options.targetD[0:1] == '-':
    print 'targetD Err, quit'
    exit()
if options.fileType == None or options.fileType[0:1] == '-':
    print 'fileType Err, quit'
    exit()
if options.workdir == None or options.workdir[0:1] == '-':
    print 'workdir Err, quit'
    exit()

# 遍历文件
for root, dirs, files in os.walk(options.srcD):
    for name in files:
        name = name.replace('[', '''\[''')  # 对文件名中的[进行转义
        newname = name[0: name.rindex('.')]
        print "Test newname: " + newname
        print "Test name: " + name

        # 运行
        cmd = 'cd ' + options.workdir + ';mkdir -p ffm;  rm -f ffm/ffm.txt ; csh -c "(ffmpeg -i ' + options.srcD + '/' + name + ' >& ffm/ffm.txt)"; grep Duration ffm/ffm.txt'
        print cmd
        (si, so, se) = os.popen3(cmd)
        t = so.readlines()
        reg = '''Duration\:\s(\d+)\:(\d+)\:([\d\.]+)'''
        duration = 0  # second
        for line in t:
            result = re.compile(reg).findall(line)
            for c in result:
                print 'split file to', options.splitsize, 'minutes, Duration:', c[0], c[1], c[2]
                duration = int(c[0]) * 3600 + int(c[1]) * 60 + float(c[2])
                nameLength = int(math.log(int(duration / (int(options.splitsize) * 60))) / math.log(10)) + 1
                print (duration)
                for i in range(int(duration / (int(options.splitsize) * 60)) + 1):
                    print i
                    _t = ''
                    if duration > int(options.splitsize) * 60 * (i + 1):
                        _t = str(int(options.splitsize) * 60)
                    else:
                        _t = str(duration - int(options.splitsize) * 60 * i)
                    cmd = 'csh -c "' + "cd " + options.workdir + ";touch ffm/output.log;(ffmpeg -y -i " + options.srcD + "/" + name + " -codec: copy -ss " + str(
                        i * int(
                            options.splitsize) * 60) + " -t " + _t + " " + options.targetD + "/" + options.prefix + newname + '_' + string.replace(
                        ('%' + str(nameLength) + 's') % str(i), ' ',
                        '0') + "." + options.fileType + ' >>& ffm/output.log)"'
                    print cmd
                    (si, so, se) = os.popen3(cmd)
                    for line in se.readlines():  # 打印输出
                        print line

