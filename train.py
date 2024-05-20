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
    parser.add_argument("--a0", type=float,
                        default=0.5, help="initial value of a")
    parser.add_argument("--b0", type=float,
                        default=0.6, help="initial value of b")
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
    ab0 = torch.tensor((args.a0, args.b0))

    # model
    model = SofaNetEllipse(ab0, hidden_sizes=args.hidden_sizes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, maximize=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    # alpha
    alpha = torch.linspace(0, torch.pi, args.n_alphas).to(args.device)

    # train
    progress_bar = trange(args.epochs)
    for epoch in progress_bar:
        xp, yp, xp_prime, yp_prime = model.forward(alpha)
        area = compute_area(alpha, xp, yp, xp_prime, yp_prime, n_area_samples=args.n_area_samples)
        optimizer.zero_grad()
        area.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix(area=f"{area.item():.4e}")

    # eval
    xp, yp, xp_prime, yp_prime = model.forward(alpha)
    area, outline = compute_area(alpha, xp, yp, xp_prime, yp_prime,
                                 n_area_samples=args.n_area_samples, return_outline=True)
    print("Maximized area:", area.item())
    plt.figure(dpi=200)
    plt.plot(outline, outline)
    plt.plot(outline, outline)
    plt.savefig("outputs/outline.png")

    # save
    torch.save(model.state_dict(), 'outputs/model_ellipse.pt')
