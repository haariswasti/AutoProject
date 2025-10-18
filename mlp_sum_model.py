import torch, torch.nn as nn, torch.optim as optim

N = 50_000
BATCH = 256
LR = 1e-3
EPOCHS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

x = (20*torch.rand(N, 2) - 10)
y = x.sum(dim=1, keepdim=True)

idx = torch.randperm(N)
tr, va = idx[:int(0.9*N)], idx[int(0.9*N):]
xtr, ytr = x[tr], y[tr]
xva, yva = x[va], y[va]

mu, sigma = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-8
xtrn = (xtr - mu) / sigma
xvan = (xva - mu) / sigma

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, z): return self.net(z)

model = MLP().to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

tr_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(xtrn, ytr), batch_size=BATCH, shuffle=True
)
va_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(xvan, yva), batch_size=1024, shuffle=False
)

for epoch in range(1, EPOCHS+1):
    model.train()
    tr_loss = 0.0
    for xb, yb in tr_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(tr_loader.dataset)

    model.eval()
    with torch.no_grad():
        va_err = 0.0
        for xb, yb in va_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            va_err += (pred - yb).abs().sum().item()
        va_mae = va_err / len(va_loader.dataset)
    print(f"epoch {epoch:02d} | train MSE {tr_loss:.6f} | val MAE {va_mae:.4f}")

with torch.no_grad():
    demo = torch.tensor([[3.5, -1.2], [10.0, 2.0]])
    demon = (demo - mu) / sigma
    out = model(demon.to(DEVICE)).cpu()
    print("inputs:\n", demo)
    print("predictions:\n", out.squeeze())
    print("true:\n", demo.sum(dim=1))
