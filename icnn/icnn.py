import time

import torch
import torch.nn.functional as F
from torch import nn, autograd
import numpy as np

from generators2d import sample_data


def to_numpy(tensor):
    return tensor.to("cpu").detach().numpy()


def to_torch(arr):
    return torch.tensor(arr, device=device, requires_grad=True)


class ICNN(torch.nn.Module):
    """Input Convex Neural Network"""

    def __init__(self, dim=2, dimh=64, num_hidden_layers=4):
        super().__init__()

        Wzs = []
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(torch.nn.Linear(dimh, dimh, bias=False))
        Wzs.append(torch.nn.Linear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = []
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)
        self.act = nn.Softplus()

    def forward(self, x):
        z = self.act(self.Wzs[0](x))
        for Wz, Wx in zip(self.Wzs[1:-1], self.Wxs[:-1]):
            z = self.act(Wz(z) + Wx(x))
        return self.Wzs[-1](z) + self.Wxs[-1](x)


def test_convexity(f):
    rdata = torch.randn(1024, 2).to(device)
    rdata2 = torch.randn(1024, 2).to(device)
    return np.all(
        (((f(rdata) + f(rdata2)) / 2 - f(rdata + rdata2) / 2) > 0)
        .cpu()
        .detach()
        .numpy()
    )


def compute_w2(f, g, x, y, return_loss=False):
    fx = f(x)
    gy = g(y)

    grad_gy = autograd.grad(torch.sum(gy), y, retain_graph=True, create_graph=True)[0]

    f_grad_gy = f(grad_gy)
    y_dot_grad_gy = torch.sum(torch.multiply(y, grad_gy), axis=1, keepdim=True)

    x_squared = torch.sum(torch.pow(x, 2), axis=1, keepdim=True)
    y_squared = torch.sum(torch.pow(y, 2), axis=1, keepdim=True)

    w2 = torch.mean(f_grad_gy - fx - y_dot_grad_gy + 0.5 * x_squared + 0.5 * y_squared)
    if not return_loss:
        return w2
    g_loss = torch.mean(f_grad_gy - y_dot_grad_gy)
    f_loss = torch.mean(fx - f_grad_gy)
    return w2, f_loss, g_loss


def plot(x, y, x_pred, y_pred, savename=None):
    x = to_numpy(x)
    y = to_numpy(y)
    x_pred = to_numpy(x_pred)
    y_pred = to_numpy(y_pred)

    import matplotlib.pyplot as plt

    plt.scatter(y[:, 0], y[:, 1], color="C1", alpha=0.5, label=r"$Y$")
    plt.scatter(x[:, 0], x[:, 1], color="C2", alpha=0.5, label=r"$X$")
    plt.scatter(
        x_pred[:, 0], x_pred[:, 1], color="C3", alpha=0.5, label=r"$\nabla g(Y)$"
    )
    plt.scatter(
        y_pred[:, 0], y_pred[:, 1], color="C4", alpha=0.5, label=r"$\nabla f(X)$"
    )
    plt.legend()
    if savename:
        plt.savefig(savename)
    plt.close()


def transport(model, x):
    return autograd.grad(torch.sum(model(x)), x)[0]


def train(f, g, x_sampler, y_sampler, batchsize=1024, reg=0):
    def y_to_x(y):
        return transport(g, y)

    def x_to_y(x):
        return transport(f, x)

    optimizer_f = torch.optim.Adam(f.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_g = torch.optim.Adam(g.parameters(), lr=1e-4, betas=(0.5, 0.9))
    nepochs = 10000
    print_interval = 100
    nval = 1024

    x_val = to_torch(next(x_sampler(nval)))
    y_val = to_torch(next(y_sampler(nval)))

    start = time.time()

    for epoch in range(1, nepochs + 1):
        for _ in range(10):
            optimizer_g.zero_grad()
            x = to_torch(next(x_sampler(batchsize)))
            y = to_torch(next(y_sampler(batchsize)))
            fx = f(x)
            gy = g(y)
            grad_gy = autograd.grad(
                torch.sum(gy), y, retain_graph=True, create_graph=True
            )[0]
            f_grad_gy = f(grad_gy)
            y_dot_grad_gy = torch.sum(torch.mul(y, grad_gy), axis=1, keepdim=True)
            g_loss = torch.mean(f_grad_gy - y_dot_grad_gy)
            if reg > 0:
                g_loss += reg * torch.sum(
                    torch.stack([torch.sum(F.relu(-w.weight) ** 2) / 2 for w in g.Wzs])
                )
            g_loss.backward()
            optimizer_g.step()

        optimizer_f.zero_grad()
        x = to_torch(next(x_sampler(batchsize)))
        y = to_torch(next(y_sampler(batchsize)))
        fx = f(x)
        gy = g(y)
        grad_gy = autograd.grad(torch.sum(gy), y, retain_graph=True, create_graph=True)[
            0
        ]
        f_grad_gy = f(grad_gy)
        f_loss = torch.mean(fx - f_grad_gy)
        if reg > 0:
            f_loss += reg * torch.sum(
                torch.stack([torch.sum(F.relu(-w.weight) ** 2) / 2 for w in f.Wzs])
            )

        f_loss.backward()
        optimizer_f.step()

        if epoch % print_interval == 0:
            w2, f_loss, g_loss = compute_w2(f, g, y_val, y_val, return_loss=True)
            end = time.time()
            print(
                f"Iter={epoch}, f_loss={f_loss:0.2f}, g_loss={g_loss:0.2f}, W2={w2:0.2f}, time={end - start:0.1f}, "
                "f_convex={test_convexity(f)}, g_convex={test_convexity(g)}"
            )
            start = end
            if plot_sample:
                x_pred = y_to_x(y_val)
                y_pred = x_to_y(x_val)
                plot(x_val, y_val, x_pred, y_pred, savename=f"tmp/epoch_{epoch}")


def main():
    scale = 5
    var = 1

    x_name = "8gaussians"
    y_name = "simpleGaussian"

    x_sampler = lambda n: sample_data(x_name, n, scale, var)
    y_sampler = lambda n: sample_data(y_name, n, scale, var)

    f = ICNN().to(device)
    g = ICNN().to(device)
    train(f, g, x_sampler, y_sampler, reg=0.1)


if __name__ == "__main__":
    plot_sample = True
    device = "cuda"
    main()
