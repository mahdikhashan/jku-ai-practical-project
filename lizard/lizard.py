import logging


LOGGER = logging.getLogger(__name__)


def lizard(
    a: int
):
    LOGGER.info(
        "custom kernel"
    )
    print(a)
    return a
