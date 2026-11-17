from __future__ import annotations

import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from src.harvest_worker import router as harvest_router
from src.voice_agent import agent_router

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "data" / "agent.log"

# When launched via pythonw.exe (no console), sys.stdout/stderr are None;
# redirect them to devnull so libraries like tqdm/print never crash.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


def configure_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
    ]
    # Only attach StreamHandler when running with a visible console (python.exe, not pythonw.exe)
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


async def run_with_boot_retry() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")

    dp = Dispatcher()
    # harvest_router handles /harvest; agent_router handles /start, /sessions, voice & text
    dp.include_router(harvest_router)
    dp.include_router(agent_router)

    retry_delay = 5
    while True:
        bot = Bot(token=token)
        try:
            logging.info("Starting CZECH_HARVESTER & Gemini Voice Agent...")
            await dp.start_polling(bot)
            break
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logging.warning(
                "Polling interrupted (%s). Retrying in %s seconds (waiting for network)...",
                exc,
                retry_delay,
            )
            await asyncio.sleep(retry_delay)
        finally:
            await bot.session.close()


def main() -> None:
    configure_logging()
    try:
        asyncio.run(run_with_boot_retry())
    except KeyboardInterrupt:
        logging.info("Agent stopped by user.")


if __name__ == "__main__":
    main()
