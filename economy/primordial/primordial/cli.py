import argparse

def generate_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="primordial: Pre-launch simulations of Great Wyrm game economy")
    parser.set_defaults(func=lambda _: parser.print_help())
    return parser

def main() -> None:
    parser = generate_argument_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
