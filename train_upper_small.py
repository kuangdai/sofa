import argparse
from pathlib import Path

import torch
from tqdm import trange

from upper.geometry import compute_area
from upper.network import SofaNet

torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("KR's upper bounds of moving sofa",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--alphas", type=float, nargs="+",
                        default=[
                            0.283794109,  # asin(7/25),
                            0.532504098,  # asin(33/65),
                            0.781214087,  # asin(119/169),
                            1.03829223,  # asin(56/65),
                            1.28700222,  # asin(24/25)
                        ], help="alphas")
    parser.add_argument("-b", "--beta1", type=float,
                        default=torch.pi / 2, help="beta1")
    parser.add_argument("-B", "--beta2", type=float,
                        default=torch.pi / 2, help="beta2")
    parser.add_argument("-n", "--n-areas", type=int,
                        default=2000, help="number of x's for area integration")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[128, 128, 128], help="hidden sizes of model")
    parser.add_argument("-l", "--lr", type=float,
                        default=1e-4, help="learning rate")
    parser.add_argument("--lr-decay-rate", type=float,
                        default=0.5, help="decay rate of lr")
    parser.add_argument("--lr-decay-step", type=int,
                        default=1000, help="decay step of lr")
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
    model = SofaNet(hidden_sizes=args.hidden_sizes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    # alpha
    alpha = torch.tensor(args.alphas, device=args.device)

    # path
    out_dir = Path(f"outputs/upper_small/{args.name}")
    out_dir.mkdir(exist_ok=True, parents=True)

    # train
    largest_area = -1.
    progress_bar = trange(args.epochs)
    for epoch in progress_bar:
        u1, u2 = model.forward(alpha)
        area = compute_area(alpha, args.beta1, args.beta2, u1, u2,
                            n_areas=args.n_areas, return_geometry=False)
        loss = -area
        if area > largest_area:
            # checkpoint best
            largest_area = area.item()
            torch.save(model.state_dict(), f"{out_dir}/best_model.pt")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix(area=f"{area.item():.4e}", largest_area=f"{largest_area:.4e}")

    # save
    u1, u2 = model.forward(alpha)
    gg = compute_area(alpha, args.beta1, args.beta2, u1, u2,
                      n_areas=args.n_areas, return_geometry=True)
    torch.save(model.state_dict(), f"{out_dir}/last_model.pt")
    torch.save(gg, f"{out_dir}/last_geometry.pt")
    print("Last area:", gg["area"].item())

    if largest_area >= 0.:
        model.load_state_dict(torch.load(f"{out_dir}/best_model.pt"))
        u1, u2 = model.forward(alpha)
        gg = compute_area(alpha, args.beta1, args.beta2, u1, u2,
                          n_areas=args.n_areas, return_geometry=True)
        torch.save(gg, f"{out_dir}/best_geometry.pt")
        print("Best area:", gg["area"].item())
