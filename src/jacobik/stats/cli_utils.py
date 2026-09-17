"""
cli_utils.py
============
Shared CLI utilities for jacobik stats scripts.
"""

import argparse
import logging
import sys


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add standard logging arguments to a parser."""
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )


def setup_logging(args: argparse.Namespace) -> None:
    """Set up the basic logger config."""
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stderr,
    )
