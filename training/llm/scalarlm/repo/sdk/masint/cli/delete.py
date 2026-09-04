from masint.api.supermassive_intelligence import SupermassiveIntelligence

import traceback

import logging

logger = logging.getLogger(__name__)


def delete(model_name):

    logger.info(f"Deleting model: {model_name}")

    try:
        smi = SupermassiveIntelligence()
        response = smi.delete(model_name)
        logger.info(f"Delete response: {response}")
    except Exception as e:
        logger.error(f"Failed to delete model: {model_name}")
        logger.error(e)
        logger.error(traceback.format_exc())

