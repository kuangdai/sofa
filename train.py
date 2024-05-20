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
    parser.add_argument("--xy-correction", action="store_true",
                        help="whether to perform xy correction")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[128, 128, 128], help="hidden sizes of model")
    parser.add_argument("--n-alphas", type=int,
                        default=10001, help="number of alphas in [0, pi]")
    parser.add_argument("--n-area-samples", type=int,
                        default=20000, help="number of x's for area calculation")
    parser.add_argument("--lr", type=float,
                        default=1e-4, help="learning rate")
    parser.add_argument("--lr-decay-rate", type=float,
                        default=0.5, help="decay rate of lr")
    parser.add_argument("--lr-decay-step", type=int,
                        default=300, help="decay step of lr")
    parser.add_argument("--epochs", type=int,
                        default=1000, help="number of epochs")
    parser.add_argument("--device", type=str,
                        default="cpu", help="training device")
    parser.add_argument("--name", type=str,
                        default="no_xy_correction", help="name for model saving")
    args = parser.parse_args()

    # a and b
    ab0 = torch.tensor((args.a0, args.b0))

    # model
    model = SofaNetEllipse(ab0, hidden_sizes=args.hidden_sizes, xy_correction=args.xy_correction).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, maximize=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    # alpha
    alpha = torch.linspace(0, torch.pi, args.n_alphas).to(args.device)

    # train
    largest_area = -1.
    progress_bar = trange(args.epochs)
    for epoch in progress_bar:
        xp, yp, xp_prime, yp_prime = model.forward(alpha)
        area = compute_area(alpha, xp, yp, xp_prime, yp_prime, n_area_samples=args.n_area_samples)
        if area > largest_area:
            # checkpoint best
            largest_area = area.item()
            torch.save(model.state_dict(), f"outputs/best_model_{args.name}.pt")
        optimizer.zero_grad()
        area.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix(area=f"{area.item():.4e}", largest_area=f"{largest_area:.4e}")

    # eval
    if largest_area >= 0.:
        model.load_state_dict(torch.load(f"outputs/best_model_{args.name}.pt"))
    xp, yp, xp_prime, yp_prime = model.forward(alpha)
    area, outline = compute_area(alpha, xp, yp, xp_prime, yp_prime,
                                 n_area_samples=args.n_area_samples, return_outline=True)
    print("Largest area found:", area.item())
    plt.figure(dpi=200)
    plt.plot(outline[0], outline[1])
    plt.plot(outline[0], outline[2])
    plt.savefig(f"outputs/outline_{args.name}.png")
