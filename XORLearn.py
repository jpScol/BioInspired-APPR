import torch
import random

# This files shows how to build a neural network using pytorch to learn the
# XOR operation

# Function to learn
def xor(a, b):
    if a == 1 and b == 1:
        return 0
    elif a == 0 and b == 0:
        return 0
    else:
        return 1

# Torch model
model = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

# Parameters
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

score = 0

steps = 1000
params = {
    'steps': 1000,
    'evaluated_steps': 250
}

for step in range(params['steps']):
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    x = torch.Tensor([a, b])

    y_pred = model(x)
    y = xor(a, b)
    
    yp = torch.Tensor([y])
    loss = loss_fn(y_pred, yp)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation
    if step >= params['steps'] - params['evaluated_steps']:
        if y == 0 and y_pred.item() < 0.5:
            score = score + 1
        if y == 1 and y_pred.item() >= 0.5:
            score = score + 1

print("Score : " + str(score) + " / " + str(params['evaluated_steps']))
