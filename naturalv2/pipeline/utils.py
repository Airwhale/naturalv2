import asyncio
import logging
import os
from typing import Any

import aiofiles
from tqdm.asyncio import tqdm


logger = logging.getLogger(__name__)


def _create_progress_bar(total: int, desc: str) -> tqdm:
    """Create a tqdm progress bar."""
    return tqdm(total=total, desc=desc, leave=False)


async def _csv_writer(
    result_queue: asyncio.Queue,
    output_filepath: str,
    pbar: tqdm,
    flush_interval: float = 5.0,
) -> int:
    """Asynchronously write results to a CSV file."""
    success_count = 0
    last_flush_time = asyncio.get_event_loop().time()
    fieldnames = None

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    try:
        async with aiofiles.open(
            output_filepath, mode="w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = None
            while True:
                try:
                    result: dict[str, Any] | False | None = await asyncio.wait_for(
                        result_queue.get(), timeout=flush_interval
                    )
                except asyncio.TimeoutError:  # Timeout reached, flush the file
                    if writer is not None:
                        await csvfile.flush()
                    continue

                try:
                    if result is None:  # Termination signal
                        logger.debug("CSV writer received termination signal.")
                        break

                    if result is False:  # Processing error
                        logger.debug("Received False result from processing.")
                        continue

                    if writer is None:
                        fieldnames = list(result.keys())
                        # Create header row
                        header_row = ["index"] + fieldnames
                        header_line = (
                            ",".join(f'"{field}"' for field in header_row) + "\n"
                        )
                        await csvfile.write(header_line)
                        writer = (
                            True  # Just use as a flag that we've written the header
                        )

                    # Create the row data
                    row_data = {"index": success_count, **result}

                    # Format the row with proper CSV quoting
                    row_values = []
                    for field in ["index"] + fieldnames:
                        value = str(row_data.get(field, ""))
                        # Escape quotes and wrap in quotes
                        escaped_value = value.replace('"', '""')
                        row_values.append(f'"{escaped_value}"')

                    row_line = ",".join(row_values) + "\n"
                    await csvfile.write(row_line)

                    success_count += 1
                    pbar.update(1)

                    # Periodic flush
                    current_time = asyncio.get_event_loop().time()
                    if (current_time - last_flush_time) >= flush_interval:
                        await csvfile.flush()
                        last_flush_time = current_time
                finally:
                    result_queue.task_done()

        # Final flush happens automatically when exiting the async context manager

    except Exception as e:
        logger.error(f"Error writing to CSV file {output_filepath}: {e}", exc_info=True)
    finally:
        pbar.close()

    return success_count
