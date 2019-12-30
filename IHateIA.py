import torch
import torch.nn as nn

def xor(a, b):
    if a == 1 and b == 1:
        return 0
    elif a == 0 and b == 0:
        return 0
    else:
        return 1

model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

import random
import numpy

score = 0

steps = 1000

for step in range(steps):
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    x = torch.Tensor([a, b])

    y_pred = model(x)
    y = xor(a, b)
    
    yp = torch.Tensor([1])
    yp[0] = y
    loss = loss_fn(y_pred, yp)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step > steps - 250:
        if y == 0 and y_pred.item() < 0.5:
            score = score + 1
        if y == 1 and y_pred.item() >= 0.5:
            score = score + 1

        print(x)
        print(y_pred)
        print(y)

print(score)
