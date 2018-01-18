# -*- coding: utf-8 -*-
# @Time    : 2018/1/19 00:32
# @Author  : DeepKeeper (DeepKeeper@qq.com)
# @Site    : 
# @File    : gpu.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# it's the key to get gpu status
import pynvml as N

# method to obtain gpu information
def get_gpu_status(gpu_index=0):
    # init for getting
    N.nvmlInit()
    handle = N.nvmlDeviceGetHandleByIndex(gpu_index)

    def _decode(b):
        if isinstance(b, bytes):
            return b.decode()  # to unicode
        return b

    name = _decode(N.nvmlDeviceGetName(handle))
    uuid = _decode(N.nvmlDeviceGetUUID(handle))

    try:
        temperature = N.nvmlDeviceGetTemperature(handle, N.NVML_TEMPERATURE_GPU)
    except N.NVMLError:
        temperature = None

    try:
        memory = N.nvmlDeviceGetMemoryInfo(handle)
    except N.NVMLError:
        memory = None

    try:
        utilization = N.nvmlDeviceGetUtilizationRates(handle)
    except N.NVMLError:
        utilization = None

    try:
        power = N.nvmlDeviceGetPowerUsage(handle)
    except:
        power = None

    try:
        power_limit = N.nvmlDeviceGetEnforcedPowerLimit(handle)
    except:
        power_limit = None

    # real gpu index
    index = N.nvmlDeviceGetIndex(handle)
    gpu_info = {
        'index': index,
        'uuid': uuid,
        'name': name,
        'temperature': temperature,
        'utilization': utilization.gpu if utilization else None,
        'power': int(power / 1000) if power is not None else None,
        'enforced.power': int(power_limit / 1000) if power_limit is not None else None,
        # Convert bytes into MBytes
        'memory.used': int(memory.used / 1024 / 1024) if memory else None,
        'memory.total': int(memory.total / 1024 / 1024) if memory else None,
    }
    # release resource
    N.nvmlShutdown()
    return gpu_info


if __name__ == '__main__':
    # {'enforced.power': 166, 'index': 0, 'utilization': 0, 'memory.used': 2
    #  175, 'uuid': 'GPU-d1468760-8b9c-9bbf-8888-f8ea61cc2a2c', 'temperature': 63,
    # 'name': 'GeForce GTX 1080', 'power': 112, 'memory.total': 8192}
    status = get_gpu_status()
    print(status)
