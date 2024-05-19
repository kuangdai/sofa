import argparse

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from src.geometry import compute_area
from src.net import SofaNetEllipse

torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pretrain ellipse",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--a-initial", type=float, nargs="+",
                        default=[0.05, 1.0, 20], help="initial values of a")
    parser.add_argument("--b-initial", type=float, nargs="+",
                        default=[0.05, 1.0, 20], help="initial values of b")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[128, 128, 128], help="hidden sizes of model")
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
    if len(args.a_initial) == 3:
        a = torch.linspace(args.a_initial[0], args.a_initial[1], int(args.a_initial[2]))
    else:
        assert len(args.a_initial) == 1
        a = torch.tensor((args.a_initial[0]))
    if len(args.b_initial) == 3:
        b = torch.linspace(args.b_initial[0], args.b_initial[1], int(args.b_initial[2]))
    else:
        assert len(args.b_initial) == 1
        b = torch.tensor((args.b_initial[0]))
    ab0 = torch.stack(torch.meshgrid(a, b, indexing="ij"), dim=-1).reshape(-1, 2).to(args.device)

    # model
    model = SofaNetEllipse(ab0, hidden_sizes=args.hidden_sizes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    # alpha
    alpha = torch.linspace(0, torch.pi, args.n_alphas).to(args.device)

    # train
    progress_bar = trange(args.epochs)
    for epoch in progress_bar:
        xp, yp, xp_prime, yp_prime = model.forward(alpha)
        area = compute_area(alpha, xp, yp, xp_prime, yp_prime, n_area_samples=args.n_area_samples)
        loss = -area.sum()  # using sum() so that nets are independently updated
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        max_area, max_loc = torch.max(area, dim=0)
        max_a, max_b = ab0[max_loc]
        progress_bar.set_postfix(area=f"{max_area.item():.4e}",
                                 ab0=f"[{max_a.item():.2f}, {max_b.item():.2f}]",
                                 index=max_loc.item())

    # eval
    xp, yp, xp_prime, yp_prime = model.forward(alpha)
    area, outline = compute_area(alpha, xp, yp, xp_prime, yp_prime,
                                 n_area_samples=args.n_area_samples, return_outline=True)
    max_area, max_loc = torch.max(area, dim=0)
    print("max area:", max_area.item())
    plt.figure(dpi=200)
    plt.plot(outline[max_loc][0], outline[max_loc][1])
    plt.plot(outline[max_loc][0], outline[max_loc][2])
    plt.savefig("outputs/outline.png")

    # save
    torch.save(model.state_dict(), 'outputs/model.pt')
