import torch


correct_value = 10.0

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print(f"Deviation of {s.dtype}: {((s - correct_value) / correct_value):.6f}")


s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(f"Deviation of {s.dtype}: {((s - correct_value) / correct_value):.6f}")


s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(f"Deviation of {s.dtype}: {((s - correct_value) / correct_value):.6f}")


s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
s += x.type(torch.float32)
print(f"Deviation of {s.dtype}: {((s - correct_value) / correct_value):.6f}")
