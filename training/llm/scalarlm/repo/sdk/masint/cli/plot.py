from masint.util.make_api_url import make_api_url
from masint.util.get_session import get_session

import matplotlib.pyplot as plt

import os
import aiohttp
import asyncio

import traceback

import logging

logger = logging.getLogger(__name__)


def plot(models, smooth, y_limit=None):

    if len(models) == 0:
        models = ["latest"]

    logger.info(f"Plotting models {models}")

    try:
        asyncio.run(plot_async(models=models, smooth=smooth, y_limit=y_limit))
    except Exception as e:
        # `model_name` is scoped to plot_async's loop, not this frame; naming it
        # here raised NameError and masked the underlying failure.
        logger.error(f"Failed to plot models {models}")
        logger.error(e)
        logger.error(traceback.format_exc())


async def plot_async(models, smooth, y_limit):
    histories = []
    model_names = []

    for model_name in models:

        status = await get_status(model_name)

        history = status["job_status"]["history"]
        model_name = clip_model_name(
            os.path.basename(status["job_config"]["job_directory"])
        )

        histories.append(history)
        model_names.append(model_name)

    # Plot loss against step
    plot_loss(histories, model_names, smooth, y_limit)


def plot_loss(histories, model_names, smooth, y_limit):

    for history, model_name in zip(histories, model_names):
        steps = [int(entry["step"]) for entry in history]
        losses = [float(entry["loss"]) for entry in history]
        losses = apply_smoothing(losses, smooth)
        plt.plot(steps, losses, label=model_name)

    model_name = model_names[0]

    # The loop above already plotted every history with a label; re-plotting the
    # last one here drew a duplicate, unlabeled line over it.
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training loss for model " + model_name)
    plt.grid(True)
    if y_limit is not None:
        plt.ylim(0, y_limit)

    base_path = "/app/cray/data"

    if os.path.isdir(base_path):
        path = f"{base_path}/loss_plot_{model_name}.pdf"
    else:
        path = f"loss_plot_{model_name}.pdf"

    logger.info(f"Saving plot to {path}")

    plt.savefig(path)


def apply_smoothing(data, smooth):
    # Number of steps to smooth over
    smooth_steps = smooth

    smoothed_data = []

    for i in range(len(data)):
        if i < smooth_steps:
            smoothed_data.append(sum(data[: i + 1]) / (i + 1))
        else:
            smoothed_data.append(sum(data[i - smooth_steps : i]) / smooth_steps)

    return smoothed_data


async def get_status(model_name):
    async with get_session() as session:
        async with session.get(make_api_url(f"v1/megatron/train/{model_name}")) as resp:
            data = await resp.json()

            logger.info(f"Got status for model {model_name}")
            logger.info(data)

            if resp.status != 200:
                logger.error(f"Failed to get status for model {model_name}")
                logger.error(data)
                raise Exception("Failed to get status for model")

            return data


def clip_model_name(model_name):
    return model_name[0:10]
