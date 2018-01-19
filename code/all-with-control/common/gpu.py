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

UNKNOWN = 'Not Supported'


class GPUStat(object):

    def __init__(self, entry):
        if not isinstance(entry, dict):
            raise TypeError('entry should be a dict, {} given'.format(type(entry)))
        self.entry = entry

        # Handle [Not Supported]
        for k in self.entry.keys():
            if isinstance(self.entry[k], str) and UNKNOWN in self.entry[k]:
                self.entry[k] = None

    def keys(self):
        return self.entry.keys()

    def __getitem__(self, key):
        return self.entry[key]

    @property
    def index(self):
        """
        Returns the index of GPU (as in nvidia-smi).
        """
        return self.entry['index']

    @property
    def uuid(self):
        """
        Returns the uuid returned by nvidia-smi,
        e.g. GPU-12345678-abcd-abcd-uuid-123456abcdef
        """
        return self.entry['uuid']

    @property
    def name(self):
        """
        Returns the name of GPU card (e.g. Geforce Titan X)
        """
        return self.entry['name']

    @property
    def memory_total(self):
        """
        Returns the total memory (in MB) as an integer
        """
        return int(self.entry['memory.total'])

    @property
    def memory_used(self):
        """
        Returns the occupied memory (in MB) as an integer
        """
        return int(self.entry['memory.used'])

    @property
    def memory_free(self):
        """
        Returns the free (available) memory (in MB) as an integer
        """
        v = self.memory_total - self.memory_used
        return max(v, 0)

    @property
    def temperature(self):
        """
        Returns the temperature of GPU as an integer
        """
        v = self.entry['temperature']
        return int(v) if v is not None else None

    @property
    def utilization(self):
        """
        Returns the GPU utilization (in percentile)
        """
        v = self.entry['utilization']
        return int(v) if v is not None else None

    @property
    def power_draw(self):
        """
        Returns the GPU power usage in Watts
        """
        v = self.entry['power']
        return int(v) if v is not None else None

    @property
    def power_limit(self):
        """
        Returns the (enforced) GPU power limit in Watts
        """
        v = self.entry['enforced.power']
        return int(v) if v is not None else None


# real method to obtain gpu information
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
    return GPUStat(gpu_info)


if __name__ == '__main__':
    # {'enforced.power': 166, 'index': 0, 'utilization': 0, 'memory.used': 2
    #  175, 'uuid': 'GPU-d1468760-8b9c-9bbf-8888-f8ea61cc2a2c', 'temperature': 63,
    # 'name': 'GeForce GTX 1080', 'power': 112, 'memory.total': 8192}
    status = get_gpu_status()
    print(status.power_limit, status.memory_used)
