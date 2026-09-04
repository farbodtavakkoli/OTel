from masint.cli.view_logs import view_logs
from masint.cli.view_service_logs import view_service_logs
from masint.cli.plot import plot
from masint.cli.ls import ls
from masint.cli.squeue import squeue
from masint.cli.stats import stats
from masint.cli.clear_queue import clear_queue
from masint.cli.cancel import cancel
from masint.cli.delete import delete

import masint

from argparse import ArgumentParser

import logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    arguments, parser = parse_arguments()

    if arguments.command == "logs":
        if arguments.service_name:
            view_service_logs(
                service_name=arguments.service_name,
                tail=arguments.tail,
                follow=arguments.follow,
                lines=arguments.lines,
            )
        else:
            # Default behavior: training job logs
            view_logs(
                model_name=arguments.model,
                tail=arguments.tail,
                lines=arguments.lines,
                follow=arguments.follow,
            )

    elif arguments.command == "plot":
        plot(models=arguments.model, smooth=int(arguments.smooth), y_limit=arguments.y_limit)

    elif arguments.command == "ls":
        ls(all=arguments.all, limit=arguments.limit)

    elif arguments.command == "squeue":
        squeue()

    elif arguments.command == "stats":
        stats()

    elif arguments.command == "clear_queue":
        clear_queue()

    elif arguments.command == "cancel":
        cancel(model_name=arguments.model)
    
    elif arguments.command == "delete":
        delete(model_name=arguments.model)

    else:
        logger.error(f"Unknown command {arguments.command}")
        parser.print_help()
        exit(1)


def setup_logging():
    logging.basicConfig(level=logging.WARNING)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def parse_arguments():
    parser = ArgumentParser(description="The command line interface for MasInt ScalarLM")

    # Add a subparser for the 'logs' command
    subparsers = parser.add_subparsers(dest="command")

    add_logs_parser(subparsers)
    add_plot_parser(subparsers)
    add_ls_parser(subparsers)
    add_squeue_parser(subparsers)
    add_stats_parser(subparsers)
    add_clear_queue_parser(subparsers)
    add_cancel_parser(subparsers)
    add_delete_parser(subparsers)

    args = parser.parse_args()

    return args, parser


def add_logs_parser(subparsers):
    logs_parser = subparsers.add_parser("logs", help="View logs")

    # Add optional positional argument for service name
    logs_parser.add_argument(
        "service_name",
        nargs="?",  # Makes it optional
        default=None,
        help="The name of the service to view logs for (optional)"
    )

    logs_parser.add_argument("--model", help="The model to view logs for", default="latest")
    logs_parser.add_argument(
        "--tail",
        help="Whether to tail the logs",
        default=False,
        action="store_true",
    )
    logs_parser.add_argument(
        "--follow",
        "-f",
        help="Whether to follow the logs",
        default=False,
        action="store_true",
    )

    logs_parser.add_argument(
        "--lines", type=int, help="The number of lines to view", default=100
    )


def add_plot_parser(subparsers):
    plot_parser = subparsers.add_parser("plot", help="Plot the results of a model")

    plot_parser.add_argument("--model", help="The model to plot results for", default=[], action="append")
    plot_parser.add_argument("--smooth", help="The number of steps to smooth over", default=1)
    plot_parser.add_argument("--y-limit", help="The y-axis limit for the plot", default=None, type=float)


def add_ls_parser(subparsers):
    ls_parser = subparsers.add_parser("ls", help="List models")
    ls_parser.add_argument("-A", "--all", help="List all attributes of the models", default=False, action="store_true")
    ls_parser.add_argument("-l", "--limit", help="Limit the number of models returned", default=None, type=int)


def add_squeue_parser(subparsers):
    squeue_parser = subparsers.add_parser("squeue", help="View the squeue")

def add_stats_parser(subparsers):
    stats_parser = subparsers.add_parser("stats", help="View the stats of the models")

def add_clear_queue_parser(subparsers):
    clear_queue_parser = subparsers.add_parser("clear_queue", help="Clear the inference queue")

def add_cancel_parser(subparsers):
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a running model")
    cancel_parser.add_argument("--model", help="The model to cancel", required=True)

def add_delete_parser(subparsers):
    delete_parser = subparsers.add_parser("delete", help="Delete a model")
    delete_parser.add_argument("--model", help="The model to delete", required=True)

if __name__ == "__main__":
    main()
