import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from accel_hydra.utils.config import (
    load_config_from_cli,
    load_config_with_overrides,
    register_omegaconf_resolvers,
)


class TestLoadConfigFromCLI(unittest.TestCase):
    """Test load_config_from_cli function"""
    def test_default_arguments(self):
        """Test default arguments"""
        with patch.object(sys, "argv", ["test_script.py"]):
            config_file, overrides = load_config_from_cli(return_config=False)
            self.assertEqual(config_file, "configs/train.yaml")
            self.assertEqual(overrides, [])

    def test_config_file_only(self):
        """Test with only config_file argument"""
        with patch.object(
            sys, "argv",
            ["test_script.py", "--config_file", "configs/test.yaml"]
        ):
            config_file, overrides = load_config_from_cli(return_config=False)
            self.assertEqual(config_file, "configs/test.yaml")
            self.assertEqual(overrides, [])

    def test_config_file_short_flag(self):
        """Test using short flag -c"""
        with patch.object(
            sys, "argv", ["test_script.py", "-c", "configs/custom.yaml"]
        ):
            config_file, overrides = load_config_from_cli(return_config=False)
            self.assertEqual(config_file, "configs/custom.yaml")
            self.assertEqual(overrides, [])

    def test_overrides_only(self):
        """Test with only overrides argument"""
        with patch.object(
            sys, "argv", [
                "test_script.py", "--overrides", "model.lr=0.001",
                "train.epochs=10"
            ]
        ):
            config_file, overrides = load_config_from_cli(return_config=False)
            self.assertEqual(config_file, "configs/train.yaml")
            self.assertEqual(overrides, ["model.lr=0.001", "train.epochs=10"])

    def test_overrides_short_flag(self):
        """Test using short flag -o"""
        with patch.object(
            sys, "argv",
            ["test_script.py", "-o", "model.lr=0.001", "train.epochs=10"]
        ):
            config_file, overrides = load_config_from_cli(return_config=False)
            self.assertEqual(config_file, "configs/train.yaml")
            self.assertEqual(overrides, ["model.lr=0.001", "train.epochs=10"])

    def test_both_arguments(self):
        """Test with both config_file and overrides"""
        with patch.object(
            sys,
            "argv",
            [
                "test_script.py",
                "--config_file",
                "configs/custom.yaml",
                "--overrides",
                "model.lr=0.001",
                "train.epochs=10",
            ],
        ):
            config_file, overrides = load_config_from_cli(return_config=False)
            self.assertEqual(config_file, "configs/custom.yaml")
            self.assertEqual(overrides, ["model.lr=0.001", "train.epochs=10"])

    def test_both_short_flags(self):
        """Test using both short flags -c and -o"""
        with patch.object(
            sys,
            "argv",
            [
                "test_script.py", "-c", "configs/test.yaml", "-o",
                "model.lr=0.001"
            ],
        ):
            config_file, overrides = load_config_from_cli(return_config=False)
            self.assertEqual(config_file, "configs/test.yaml")
            self.assertEqual(overrides, ["model.lr=0.001"])

    def test_mixed_flags(self):
        """Test mixing long and short flags"""
        with patch.object(
            sys,
            "argv",
            [
                "test_script.py", "-c", "configs/test.yaml", "--overrides",
                "model.lr=0.001"
            ],
        ):
            config_file, overrides = load_config_from_cli(return_config=False)
            self.assertEqual(config_file, "configs/test.yaml")
            self.assertEqual(overrides, ["model.lr=0.001"])

    def test_empty_overrides(self):
        """Test with explicitly empty overrides"""
        with patch.object(sys, "argv", ["test_script.py", "--overrides"]):
            config_file, overrides = load_config_from_cli(return_config=False)
            self.assertEqual(config_file, "configs/train.yaml")
            self.assertEqual(overrides, [])

    def test_multiple_overrides(self):
        """Test with multiple overrides"""
        with patch.object(
            sys,
            "argv",
            [
                "test_script.py",
                "--overrides",
                "model.lr=0.001",
                "model.batch_size=32",
                "train.epochs=10",
                "train.optimizer=adam",
            ],
        ):
            config_file, overrides = load_config_from_cli(return_config=False)
            self.assertEqual(config_file, "configs/train.yaml")
            self.assertEqual(len(overrides), 4)
            self.assertIn("model.lr=0.001", overrides)
            self.assertIn("model.batch_size=32", overrides)
            self.assertIn("train.epochs=10", overrides)
            self.assertIn("train.optimizer=adam", overrides)

    def test_load_config_file_from_cli(self):
        """Test load_config_from_cli with actual YAML file"""
        config_file = str(
            Path(__file__).parent / "configs" / "test_simple.yaml"
        )
        with patch.object(
            sys, "argv", [
                "test_script.py",
                "-c",
                config_file,
                "-o",
                "seed=1",
                "model.hidden_dim=512",
            ]
        ):
            config = load_config_from_cli()

            self.assertEqual(
                config, {
                    "seed": 1,
                    "exp_name": "test",
                    "model": {
                        "_target_": "test_model.DummyModel",
                        "hidden_dim": 512
                    }
                }
            )


class TestLoadConfigWithOverrides(unittest.TestCase):
    """Test load_config_from_cli and load_config_with_overrides with actual YAML files"""
    def setUp(self):
        """Set up test fixtures"""
        self.test_configs_dir = Path(__file__).parent / "configs"

    def test_load_simple_config(self):
        """Test loading a simple YAML config file"""
        config_file = self.test_configs_dir / "test_simple.yaml"
        config = load_config_with_overrides(str(config_file), [])

        self.assertEqual(
            config, {
                "seed": 42,
                "exp_name": "test",
                "model": {
                    "_target_": "test_model.DummyModel",
                    "hidden_dim": 256
                }
            }
        )

    def test_load_config_with_overrides(self):
        """Test loading config with command line overrides"""
        config_file = self.test_configs_dir / "test_simple.yaml"
        overrides = ["seed=1", "model.hidden_dim=512"]
        config = load_config_with_overrides(str(config_file), overrides)

        self.assertEqual(
            config, {
                "seed": 1,
                "exp_name": "test",
                "model": {
                    "_target_": "test_model.DummyModel",
                    "hidden_dim": 512
                }
            }
        )

    def test_load_config_with_interpolation(self):
        """Test loading config with variable interpolation"""
        config_file = self.test_configs_dir / "test_interpolation.yaml"
        config = load_config_with_overrides(str(config_file), [])

        self.assertEqual(config["trainer"]["epochs"], config["epochs"])
        self.assertEqual(config["trainer"]["exp_name"], config["exp_name"])
        self.assertEqual(
            config["model"]["hidden_dim"],
            config["model"]["feature_extractor"]["out_dim"]
        )

    def test_load_config_with_resolvers(self):
        """Test loading config with OmegaConf resolvers"""
        config_file = self.test_configs_dir / "test_with_resolver.yaml"
        config = load_config_with_overrides(str(config_file), [])

        # Test len resolver
        self.assertEqual(config["vocab_size"], 5)

        # Test multiply resolver
        self.assertEqual(
            config["total_params"], config["num_layers"] * config["hidden_dim"]
        )


class TestCustomRegisterResolverFn(unittest.TestCase):
    """Test load_config_from_cli and load_config_with_overrides with custom register_resolver_fn"""
    def setUp(self):
        """Set up test fixtures"""
        self.test_configs_dir = Path(__file__).parent / "configs"

    def test_extend_default_resolvers(self):
        def get_warmup_steps(epochs: int, epoch_length: int):
            return epochs * epoch_length * 0.01

        def custom_register_resolvers():
            register_omegaconf_resolvers()
            OmegaConf.register_new_resolver(
                "get_warmup_steps", get_warmup_steps, replace=True
            )

        config = load_config_with_overrides(
            str(self.test_configs_dir / "test_custom_resolver.yaml"), [],
            register_resolver_fn=custom_register_resolvers
        )

        self.assertEqual(
            config["warmup_steps"],
            config["epochs"] * config["epoch_length"] * 0.01
        )


if __name__ == "__main__":
    unittest.main()
