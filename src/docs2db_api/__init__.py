import logging

import structlog

from docs2db_api.config import settings

# Get log level from Pydantic settings
log_level_str = settings.logging.log_level.upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(level=logging.WARNING)

# Configure structlog for beautiful console output
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(log_level),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)
