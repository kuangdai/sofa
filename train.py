import argparse

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from src.geometry import compute_area
from src.net import SofaNetEllipse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pretrain ellipse",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--a-linspace", type=float, nargs="+",
                        default=[0.05, 1.0, 20], help="linear space of a")
    parser.add_argument("--b-linspace", type=float, nargs="+",
                        default=[0.05, 1.0, 20], help="linear space of b")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[128, 128], help="hidden sizes of model")
    parser.add_argument("--n-alphas", type=int,
                        default=1001, help="number of alphas in [0, pi]")
    parser.add_argument("--n-area-samples", type=int,
                        default=2000, help="number of x's for area calculation")
    parser.add_argument("--lr", type=float,
                        default=0.0001, help="learning rate")
    parser.add_argument("--lr-decay-rate", type=float,
                        default=0.5, help="decay rate of lr")
    parser.add_argument("--lr-decay-step", type=int,
                        default=1000, help="decay step of lr")
    parser.add_argument("--epochs", type=int,
                        default=10000, help="number of epochs")
    parser.add_argument("--device", type=str,
                        default="cpu", help="training device")
    args = parser.parse_args()

    # a and b
    a = torch.linspace(args.a_linspace[0], args.a_linspace[1], int(args.a_linspace[2]))
    b = torch.linspace(args.b_linspace[0], args.b_linspace[1], int(args.b_linspace[2]))
    ab = torch.stack(torch.meshgrid(a, b, indexing="ij"), dim=-1).reshape(-1, 2).to(args.device)

    # model
    model = SofaNetEllipse(ab, hidden_sizes=args.hidden_sizes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    # alpha
    alpha = torch.linspace(0, torch.pi, args.n_alphas).to(args.device)

    # train
    progress_bar = trange(args.epochs)
    for epoch in progress_bar:
        a, b, da, db = model.forward(alpha, compute_gradients=True)
        area = compute_area(alpha, a, b, da, db, n_area_samples=args.n_area_samples)
        loss = -area.sum()  # using sum() so that nets are independently updated
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        max_area, max_loc = torch.max(area, dim=0)
        max_a, max_b = ab[max_loc]
        progress_bar.set_postfix(area=f"{max_area.item():.4e}", ab=f"[{max_a.item():.2f}, {max_b.item():.2f}]")

    # eval
    a, b, da, db = model.forward(alpha, compute_gradients=True)
    area, outline = compute_area(alpha, a, b, da, db, n_area_samples=args.n_area_samples, return_outline=True)
    max_area, max_loc = torch.max(area, dim=0)
    print(max_area.item())
    plt.figure(dpi=200)
    plt.plot(outline[max_loc][:, 0], outline[max_loc][:, 1])
    plt.savefig("outputs/outline.png")

    # save
    torch.save(model.state_dict(), 'outputs/model.pt')
