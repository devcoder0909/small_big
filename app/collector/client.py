"""HTTP client for fetching data from the WinGo source API."""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from app.core import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FetchResult:
    """Result of a source API fetch."""

    success: bool
    status_code: int | None
    data: dict | None
    response_time_ms: int
    request_timestamp_ms: int
    requested_at: datetime
    error_type: str | None = None
    error_message: str | None = None


class SourceClient:
    """Async HTTP client for the WinGo source API."""

    def __init__(self):
        settings = get_settings()
        self.source_url = settings.source_url
        self.max_retries = settings.max_retries
        self.backoff_base = settings.backoff_base_seconds
        self.backoff_max = settings.backoff_max_seconds
        self.timeout = httpx.Timeout(
            connect=5.0,
            read=float(settings.request_timeout_seconds),
            write=5.0,
            pool=5.0,
        )
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self.headers,
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def fetch_history(self) -> FetchResult:
        """
        Fetch the latest history page from the source API.

        Generates a dynamic timestamp for cache-busting.
        Uses exponential backoff retries for network resilience.
        """
        ts = int(time.time() * 1000)
        requested_at = datetime.now(timezone.utc)
        start_time = time.monotonic()

        client = await self._get_client()
        last_error = None
        last_status = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.get(
                    self.source_url,
                    params={"ts": ts},
                )
                last_status = response.status_code

                if response.status_code == 200:
                    elapsed_ms = int((time.monotonic() - start_time) * 1000)
                    data = response.json()

                    logger.info(
                        "source_request_success",
                        status_code=200,
                        response_time_ms=elapsed_ms,
                        attempt=attempt,
                        ts=ts,
                    )

                    return FetchResult(
                        success=True,
                        status_code=200,
                        data=data,
                        response_time_ms=elapsed_ms,
                        request_timestamp_ms=ts,
                        requested_at=requested_at,
                    )

                logger.warning(
                    "source_request_non_200",
                    status_code=response.status_code,
                    attempt=attempt,
                )

            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                last_error = e
                logger.warning(
                    "source_request_network_retry",
                    attempt=attempt,
                    error=str(e),
                    error_type=type(e).__name__,
                )

            except Exception as e:
                last_error = e
                logger.error(
                    "source_request_unexpected",
                    attempt=attempt,
                    error=str(e),
                )
                break

            # Backoff before retry
            if attempt < self.max_retries:
                delay = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
                await asyncio.sleep(delay)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        err_msg = str(last_error) if last_error else f"HTTP Status {last_status}"
        err_type = type(last_error).__name__ if last_error else "HTTPError"

        return FetchResult(
            success=False,
            status_code=last_status,
            data=None,
            response_time_ms=elapsed_ms,
            request_timestamp_ms=ts,
            requested_at=requested_at,
            error_type=err_type,
            error_message=err_msg,
        )
