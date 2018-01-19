# -*- coding: utf-8 -*-
# @Time    : 2018/1/18 23:05
# @Author  : DeepKeeper (DeepKeeper@qq.com)
# @Site    : 
# @File    : cpu-ram.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import platform

import psutil


# memory usage in percent
def get_ram_status():
    mem_stat = psutil.virtual_memory()
    return '{:.1%}'.format((mem_stat.total - mem_stat.free)/mem_stat.total)
    # return '{}%'.format(mem_stat.percent)


# get server load status
def get_server_load():
    # server load average, server's up-time
    up_time = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
    # little trick on windows,cuz windows does not support getloadavg()
    av1, av2, av3 = 0.1, 0.1, 0.1
    if platform.system() == 'Windows':
        load = '{:.1%}'.format(psutil.cpu_percent(interval=0.2)/100)
    else:
        av1, av2, av3 = os.getloadavg()
        load = "%.2f %.2f %.2f " %(av1, av2, av3)
    up_time = str(up_time).split('.')[0]
    return load, up_time


if __name__ == '__main__':
    free_ram = get_ram_status()
    print(free_ram)
    server_load, time = get_server_load()
    print(server_load, time)

