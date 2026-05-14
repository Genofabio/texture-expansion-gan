from datetime import datetime
import torch
import gc

def count_tensors():
    count_gpu = 0
    count_cpu = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    count_gpu += 1
                else:
                    count_cpu += 1
        except Exception:
            pass
    return count_cpu, count_gpu


def count_cpu_tensors():
    return sum(1 for obj in gc.get_objects() if torch.is_tensor(obj) and obj.device.type == 'cpu')