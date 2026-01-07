import hydra

from .train_launcher import TrainLauncher
from .utils.config import parse_launch_args

if __name__ == "__main__":
    launcher_class = parse_launch_args()
    launcher: TrainLauncher = hydra.utils.get_class(launcher_class)()
    launcher.run()
