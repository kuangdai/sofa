import argparse

import torch
from tqdm import trange

from src.geometry import compute_area
from src.network import SofaNet

torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Moving sofa",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--n-times", type=int,
                        default=1000, help="number of times")
    parser.add_argument("-a", "--n-areas", type=int,
                        default=2000, help="number of x's for area integration")
    parser.add_argument("--beta-deg", type=float,
                        default=81.0, help="minimum rotation angle in degree")
    parser.add_argument("--beta-factor", type=float,
                        default=5.0, help="factor for beta inequality loss")
    parser.add_argument("-E", "--envelope", action="store_true",
                        help="consider envelope when computing area")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[128, 128, 128], help="hidden sizes of model")
    parser.add_argument("-l", "--lr", type=float,
                        default=1e-4, help="learning rate")
    parser.add_argument("--lr-decay-rate", type=float,
                        default=1., help="decay rate of lr")
    parser.add_argument("--lr-decay-step", type=int,
                        default=1000, help="decay step of lr")
    parser.add_argument("-e", "--epochs", type=int,
                        default=3000, help="number of epochs")
    parser.add_argument("-d", "--device", type=str,
                        default="cpu", help="training device")
    parser.add_argument("-n", "--name", type=str,
                        default="recent", help="name of running")
    parser.add_argument("-s", "--seed", type=int,
                        default=0, help="random seed")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # model
    model = SofaNet(hidden_sizes=args.hidden_sizes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    # time
    t = torch.linspace(0., 1., args.n_times)

    # train
    beta = torch.deg2rad(torch.tensor(args.beta_deg, device=args.device))
    largest_area = -1.
    progress_bar = trange(args.epochs)
    for epoch in progress_bar:
        alpha, xp, yp, dt_alpha, dt_xp, dt_yp = model.forward(t)
        area = compute_area(t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp,
                            n_areas=args.n_areas, envelope=args.envelope, return_geometry=False)
        loss = -area + args.beta_factor * torch.relu(beta - alpha[-1])
        if area > largest_area:
            # checkpoint best
            largest_area = area.item()
            torch.save(model.state_dict(), f"outputs/best_model_{args.name}.pt")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix(area=f"{area.item():.4e}", largest_area=f"{largest_area:.4e}")

    # save
    alpha, xp, yp, dt_alpha, dt_xp, dt_yp = model.forward(t)
    area, gg = compute_area(t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp,
                            n_areas=args.n_areas, envelope=args.envelope, return_geometry=True)
    torch.save(model.state_dict(), f"outputs/last_model_{args.name}.pt")
    torch.save(gg, f"outputs/last_geometry_{args.name}.pt")
    print("Last area:", area.item())

    if largest_area >= 0.:
        model.load_state_dict(torch.load(f"outputs/best_model_{args.name}.pt"))
        alpha, xp, yp, dt_alpha, dt_xp, dt_yp = model.forward(t)
        area, gg = compute_area(t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp,
                                n_areas=args.n_areas, envelope=args.envelope, return_geometry=True)
        torch.save(gg, f"outputs/best_geometry_{args.name}.pt")
        print("Best area:", area.item())
