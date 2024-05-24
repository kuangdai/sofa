import argparse

import torch
from tqdm import trange

from src.geometry import compute_area
from src.net import SofaNet

torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Moving sofa",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--velocity-mode", action="store_true",
                        help="velocity mode")
    parser.add_argument("-t", "--n-times", type=int,
                        default=1000, help="number of times")
    parser.add_argument("-a", "--n-areas", type=int,
                        default=2000, help="number of x's for area integration")
    parser.add_argument("--a0", type=float,
                        default=0.5, help="a0 for path initialization")
    parser.add_argument("--b0", type=float,
                        default=0.6, help="b0 for path initialization")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[128, 128, 128], help="hidden sizes of model")
    parser.add_argument("-l", "--lr", type=float,
                        default=1e-4, help="learning rate")
    parser.add_argument("--lr-decay-rate", type=float,
                        default=0.5, help="decay rate of lr")
    parser.add_argument("--lr-decay-step", type=int,
                        default=300, help="decay step of lr")
    parser.add_argument("-e", "--epochs", type=int,
                        default=1000, help="number of epochs")
    parser.add_argument("-d", "--device", type=str,
                        default="cpu", help="training device")
    parser.add_argument("-n", "--name", type=str,
                        default="last", help="name of running")
    parser.add_argument("-s", "--seed", type=int,
                        default=0, help="random seed")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # model
    model = SofaNet(n_in=args.n_times, n_t=args.n_times, hidden_sizes=args.hidden_sizes,
                    a0=args.a0, b0=args.b0, velocity_mode=args.velocity_mode).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, maximize=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    # train
    largest_area = -1.
    progress_bar = trange(args.epochs)
    for epoch in progress_bar:
        t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp = model.forward()
        area = compute_area(t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp, n_areas=args.n_areas)
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
    t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp = model.forward()
    area, gg = compute_area(t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp,
                            n_areas=args.n_areas, return_geometry=True)
    torch.save(gg, f"outputs/best_geometry_{args.name}.pt")
    print("Largest area found:", area.item())
