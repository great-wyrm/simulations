import argparse
from dataclasses import asdict
import json
import os

import joblib

from . import game


def handle_sim(args: argparse.Namespace) -> None:
    if args.load:
        session = joblib.load(args.config)
    else:
        configuration = joblib.load(args.config)
        session = game.GameSession(configuration, args.population_growth_interval)

    session.rollout(args.ticks, args.shuffle, not args.quiet)

    if args.outfile is not None:
        joblib.dump(session, args.outfile)


def handle_config_init(args: argparse.Namespace) -> None:
    if os.path.exists(args.config):
        print("Configuration file already exists")
    else:
        joblib.dump([], args.config)


def handle_config_view(args: argparse.Namespace) -> None:
    if os.path.exists(args.configuration):
        configuration = joblib.load(args.configuration)
        data = [asdict(nation) for nation in configuration]
        print(json.dumps(data, indent=2))
    else:
        print("Configuration file not found")


def handle_config_nation(args: argparse.Namespace) -> None:
    configuration = []
    if os.path.exists(args.config):
        configuration = joblib.load(args.config)

    nation = game.generate_variants(
        mentality_alpha=(args.builder, args.hoarder, args.trader),
        parameter_alpha=(
            args.aggression,
            args.caution,
            args.efficiency,
            args.fertility,
        ),
        num_variants=1,
        starting_population=args.starting_population,
        starting_resources=args.starting_resources,
        starting_technology=args.starting_technology,
    )[0]

    configuration.append(nation)

    joblib.dump(configuration, args.config)


def generate_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="primordial: Pre-launch simulations of Great Wyrm game economy"
    )
    parser.set_defaults(func=lambda _: parser.print_help())
    subparsers = parser.add_subparsers()

    config_parser = subparsers.add_parser(
        "config", help="Manage simulation configurations"
    )
    config_parser.set_defaults(func=lambda _: config_parser.print_help())
    config_subparsers = config_parser.add_subparsers()

    sim_parser = subparsers.add_parser("sim", help="Run a simulation")
    sim_parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Configuration file for the simulation, which specifies the nations which are going to participate",
    )
    sim_parser.add_argument(
        "--population-growth-interval",
        type=int,
        required=False,
        default=10,
        help="Number of ticks between population growth",
    )
    sim_parser.add_argument(
        "--load",
        action="store_true",
        help="If this flag is set, it means that the configuration file is actually a saved game session",
    )
    sim_parser.add_argument(
        "-o",
        "--outfile",
        required=False,
        default=None,
        help="Save the results of the simulation to this path",
    )
    sim_parser.add_argument(
        "--ticks",
        required=True,
        type=int,
        help="Number of ticks to run the simulation for",
    )
    sim_parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If this flag is set, shuffle nations on every tick",
    )
    sim_parser.add_argument(
        "--quiet",
        action="store_true",
        help="If this flag is set, don't print anything to stdout",
    )
    sim_parser.set_defaults(func=handle_sim)

    config_init_usage = "Initialize a new configuration file"
    config_init_parser = config_subparsers.add_parser(
        "init", help=config_init_usage, description=config_init_usage
    )
    config_init_parser.add_argument(
        "-c", "--config", required=True, help="Configuration file to initialize"
    )
    config_init_parser.set_defaults(func=handle_config_init)

    config_nation_usage = "Add a nation to a configuration file"
    config_nation_parser = config_subparsers.add_parser(
        "nation", help=config_nation_usage, description=config_nation_usage
    )
    config_nation_parser.add_argument(
        "-c", "--config", required=True, help="Configuration file to modify"
    )
    config_nation_parser.add_argument(
        "--builder", required=True, type=float, help='Weight of "builder" mentality'
    )
    config_nation_parser.add_argument(
        "--hoarder", required=True, type=float, help='Weight of "hoarder" mentality'
    )
    config_nation_parser.add_argument(
        "--trader", required=True, type=float, help='Weight of "trader" mentality'
    )
    config_nation_parser.add_argument(
        "--aggression",
        required=True,
        type=float,
        help='Weight of "aggression" parameter',
    )
    config_nation_parser.add_argument(
        "--caution",
        required=True,
        type=float,
        help='Weight of "caution" parameter',
    )
    config_nation_parser.add_argument(
        "--efficiency",
        required=True,
        type=float,
        help='Weight of "efficiency" parameter',
    )
    config_nation_parser.add_argument(
        "--fertility",
        required=True,
        type=float,
        help='Weight of "fertility" parameter',
    )
    config_nation_parser.add_argument(
        "--starting-population",
        type=int,
        required=False,
        default=10,
        help="Starting population of nation",
    )
    config_nation_parser.add_argument(
        "--starting-resources",
        type=int,
        required=False,
        default=100,
        help="Total number of resources available to nation at start (will be divided across resource types in proportion to parameters of the nation)",
    )
    config_nation_parser.add_argument(
        "--starting-technology",
        type=int,
        required=False,
        default=1,
        help="Starting technology level of nation",
    )
    config_nation_parser.set_defaults(func=handle_config_nation)

    config_view_usage = "View the contents of a configuration file"
    config_view_parser = config_subparsers.add_parser(
        "view", help=config_view_usage, description=config_view_usage
    )
    config_view_parser.add_argument(
        "-c", "--configuration", required=True, help="Configuration file to view"
    )
    config_view_parser.set_defaults(func=handle_config_view)

    return parser


def main() -> None:
    parser = generate_argument_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
