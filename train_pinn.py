import argparse
from pathlib import Path

import torch
from tqdm import trange

from pinn.geometry import compute_area
from pinn.network import SofaNet

torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Moving sofa with a PINN",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--n-times", type=int,
                        default=1000, help="number of times")
    parser.add_argument("-n", "--n-areas", type=int,
                        default=2000, help="number of x's for area integration")
    parser.add_argument("--beta-deg", type=float,
                        default=81.0, help="minimum rotation angle in degree")
    parser.add_argument("--beta-factor", type=float,
                        default=5.0, help="factor for beta inequality loss")
    parser.add_argument("-E", "--envelope", action="store_true",
                        help="consider envelope when computing area")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[128, 128, 128], help="hidden sizes of model")
    parser.add_argument("--tanh", action="store_true",
                        help="use tanh for activation")
    parser.add_argument("--scaling", type=float, nargs=3,
                        default=[1., 1., 1.], help="scaling network outputs for alpha, xp, yp")
    parser.add_argument("-l", "--lr", type=float,
                        default=1e-4, help="learning rate")
    parser.add_argument("--lr-decay-rate", type=float,
                        default=0.5, help="decay rate of lr")
    parser.add_argument("--lr-decay-step", type=int,
                        default=1000, help="decay step of lr")
    parser.add_argument("-w", "--weight-decay", type=float,
                        default=0., help="weight decay")
    parser.add_argument("-e", "--epochs", type=int,
                        default=5000, help="number of epochs")
    parser.add_argument("-d", "--device", type=str,
                        default="cpu", help="training device")
    parser.add_argument("-N", "--name", type=str,
                        default="recent", help="name of running")
    parser.add_argument("-s", "--seed", type=int,
                        default=0, help="random seed")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # model
    model = SofaNet(hidden_sizes=args.hidden_sizes, tanh=args.tanh,
                    alpha_scaling=args.scaling[0],
                    xp_scaling=args.scaling[1],
                    yp_scaling=args.scaling[2]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    # time
    t = torch.linspace(0., 1., args.n_times, device=args.device)

    # path
    out_dir = Path(f"outputs/pinn/{args.name}")
    out_dir.mkdir(exist_ok=True, parents=True)

    # train
    beta = torch.deg2rad(torch.tensor(args.beta_deg, device=args.device))
    largest_area = -1.
    progress_bar = trange(args.epochs)
    for epoch in progress_bar:
        alpha, xp, yp, dt_alpha, dt_xp, dt_yp = model.forward(t)
        area = compute_area(t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp,
                            n_areas=args.n_areas, envelope=args.envelope, return_geometry=False)
        loss = -area + args.beta_factor * torch.relu(beta - alpha[-1])
        if area.item() > largest_area:
            # checkpoint best
            largest_area = area.item()
            torch.save(model.state_dict(), f"{out_dir}/best_model.pt")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix(area=f"{area.item():.5e}", largest_area=f"{largest_area:.5e}")

    # save
    alpha, xp, yp, dt_alpha, dt_xp, dt_yp = model.forward(t)
    gg = compute_area(t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp,
                      n_areas=args.n_areas, envelope=args.envelope, return_geometry=True)
    torch.save(model.state_dict(), f"{out_dir}/last_model.pt")
    torch.save(gg, f"{out_dir}/last_geometry.pt")
    print("Last area:", gg["area"].item())

    if largest_area >= 0.:
        model.load_state_dict(torch.load(f"{out_dir}/best_model.pt"))
        alpha, xp, yp, dt_alpha, dt_xp, dt_yp = model.forward(t)
        gg = compute_area(t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp,
                          n_areas=args.n_areas, envelope=args.envelope, return_geometry=True)
        torch.save(gg, f"{out_dir}/best_geometry.pt")
        print("Best area:", gg["area"].item())
