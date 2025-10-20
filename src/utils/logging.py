import logging

import click
from colorama import Fore, Style


class ColorFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        if levelname == "INFO":
            record.msg = record.msg
        elif levelname == "ERROR":
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
        elif levelname == "WARNING":
            record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter())
logger.addHandler(handler)
logger.propagate = False


# use click.echo to print to the console
def echo(message):
    click.echo(message)


def success(message):
    echo(f"{Fore.GREEN}{message}{Style.RESET_ALL}")


def error(message):
    echo(f"{Fore.RED}{message}{Style.RESET_ALL}")


def warning(message):
    echo(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")
