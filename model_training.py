from argparse import ArgumentParser
from enum import Enum

from hyperparameters import *
from train import train, save_model, plotting_multiple_models
from models import *
from datasets import *


class Cmd(Enum):
    per_frame = ("per-frame", get_single_frame_model)
    late_fusion = ("late-fusion", get_late_fusion_model)
    early_fusion = ("early-fusion", None)
    CNN_3D = ("CNN-3D", None)
    dual_stream = ("dual-stream", None)

    @classmethod
    def with_training_args(cls) -> set["Cmd"]:
        return {
            cls.per_frame,
            cls.late_fusion,
            cls.early_fusion,
            cls.CNN_3D,
            cls.dual_stream,
        }

    @property
    def func(self):
        return self.value[1]

    @property
    def mode(self):
        return self.value[0]


def add_default_args(parser: ArgumentParser):
    """Attach default training arguments to a parser."""
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("--lr", default=LEARNING_RATE, type=float)
    parser.add_argument("--plot_name", default="", type=str)
    parser.add_argument("--plot_title", default="", type=str)
    return parser


def parsing():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Common parent parser for data-related options
    data_parent = ArgumentParser(add_help=False)
    data_parent.add_argument("--leakage", action="store_true", help="Enable data leakage flag")

    # Define which commands should include training args

    for cmd in Cmd:
        parents = [data_parent]
        sub = subparsers.add_parser(cmd.mode, parents=parents, help=f"Run {cmd.mode} mode")

        if cmd in Cmd.with_training_args():
            add_default_args(sub)

        sub.set_defaults(cmd=cmd)

    return parser.parse_args()

def training_overview(out_dict):
    for model_performance, name in out_dict:
        name = f'[{name}]'
        print(f"{name:<30}"
              f"\t Epochs: {args.epochs}"
              f"\t Loss train: {model_performance['train_loss'][-1]:.3f}"
              f"\t test: {model_performance['test_loss'][-1]:.3f}"
              f"\t Accuracy train: {model_performance['train_acc'][-1] * 100:.1f}%"
              f"\t test: {model_performance['test_acc'][-1] * 100:.1f}%")

def training(model_list, device):
    performance = []

    for model in model_list:
        print(f'Training on {model.name}')

        model.to(device)

        (train_loader, test_loader, val_loader), (trainset, testset, valtest) = datasetVideoStackFrames()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        out_dict = train(model, optimizer, trainset, train_loader, valtest, val_loader, device=device,
                         num_epochs=args.epochs)

        performance.append((out_dict, model.name))

    training_overview(performance)

    return performance


def save_plot(figure, name, title, args):
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f'Training results of {args.cmd.mode} with {args.epochs} epochs')
    fig.tight_layout()

    figure.savefig(f"figures/{name}.png")
    figure.savefig(f"figures/{name}.pdf")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parsing()

    cmd = args.cmd
    print(f'Training in [{cmd.mode}] mode')

    models = cmd.func()

    performance = training(model_list=models, device=device)

    fig = plotting_multiple_models(performance)
    if args.plot_name:
        save_plot(fig, args.plot_name, args.plot_title, args)
    else:
        save_plot(fig, f'{cmd.mode}_{args.epochs}', args.plot_title, args)
