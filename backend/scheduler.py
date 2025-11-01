"""Entry point to boot the APScheduler-based MLOps simulation."""

from __future__ import annotations

import logging
import time

from src import mlops_loop


LOGGER = logging.getLogger(__name__)


def start_scheduler() -> None:
    """Kick off the background scheduler for weekly scoring."""

    scheduler = mlops_loop.schedule_weekly_job()
    LOGGER.info("Scheduler started with jobs: %s", scheduler.get_jobs())

    try:
        while True:
            scheduler.print_jobs()
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover - manual run only.
        scheduler.shutdown()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    start_scheduler()

