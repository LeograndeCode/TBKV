from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


def get_cli_config(config_location="."):
    # Parse command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "name",
        help=f'the configuration name (the file is "{config_location}/<name>.yml")',
    )
    parser.add_argument(
        "overrides", nargs="*", help="configuration overrides (like a.b.c=value)"
    )
    args = parser.parse_args()

    # Merge the configuration file and command-line overrides.
    # Support either a config name (legacy) or a direct .yml path.
    requested = Path(args.name)
    if requested.is_file():
        config_path = requested
    else:
        config_path = Path(config_location, f"{args.name}.yml")
    config = load_config(config_path, to_container=False)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    # Generate a unique name for this configuration.
    if "_name" not in config:
        if len(args.overrides) == 0:
            name = config_path.stem
        else:
            name = f"{config_path.stem}-{'-'.join(args.overrides)}"
        config["_name"] = name

    return OmegaConf.to_container(config, resolve=True)


def initialize_run(config_location="."):
    config = get_cli_config(config_location=config_location)

    if "_output" in config:
        # Create an output directory and save the merged configuration.
        output_dir = Path(config["_output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, output_dir / "config.yml", resolve=True)

    return config


def load_config(config_path, to_container=True):
    # Load the specified config, composing with defaults if necessary.
    config = OmegaConf.load(config_path)
    defaults = []
    for defaults_path in config.pop("_defaults", []):
        relative_path = Path(config_path).parent / defaults_path
        cwd_path = Path(defaults_path)
        repo_path = REPO_ROOT / defaults_path
        if relative_path.is_file():
            chosen_path = relative_path
        elif cwd_path.is_file():
            chosen_path = cwd_path
        elif repo_path.is_file():
            chosen_path = repo_path
        else:
            chosen_path = defaults_path
        defaults.append(load_config(chosen_path, to_container=False))
    config = OmegaConf.merge(*defaults, config)
    return OmegaConf.to_container(config, resolve=True) if to_container else config
