"""HTTP client for fetching data from the WinGo source API with multi-endpoint failover."""

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
    used_endpoint: str | None = None


class SourceClient:
    """Async HTTP client for the WinGo source API with redundant failover endpoints."""

    def __init__(self):
        settings = get_settings()
        self.primary_url = settings.source_url
        self.fallback_urls = [
            settings.source_url,
            settings.source_api_url,
            "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
        ]
        # Deduplicate while preserving order
        self.endpoints = list(dict.fromkeys(self.fallback_urls))

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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
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

    async def fetch_history(
        self, page_no: int = 1, page_size: int = 20
    ) -> FetchResult:
        """
        Fetch a history page from the source API with multi-endpoint failover.

        Args:
            page_no: Page number to request (default: 1).
            page_size: Page size to request (default: 20).

        Returns:
            FetchResult with status and parsed JSON dictionary payload.
        """
        ts = int(time.time() * 1000)
        requested_at = datetime.now(timezone.utc)
        start_time = time.monotonic()

        client = await self._get_client()
        last_error = None
        last_status = None

        for endpoint in self.endpoints:
            params = {"ts": ts}
            if page_no > 1 or page_size != 10:
                params["pageNo"] = str(page_no)
                params["pageSize"] = str(page_size)

            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.get(endpoint, params=params)
                    last_status = response.status_code

                    if response.status_code == 200:
                        elapsed_ms = int((time.monotonic() - start_time) * 1000)
                        data = response.json()

                        logger.info(
                            "source_request_success",
                            endpoint=endpoint,
                            status_code=200,
                            response_time_ms=elapsed_ms,
                            attempt=attempt,
                            ts=ts,
                            page_no=page_no,
                        )

                        return FetchResult(
                            success=True,
                            status_code=200,
                            data=data,
                            response_time_ms=elapsed_ms,
                            request_timestamp_ms=ts,
                            requested_at=requested_at,
                            used_endpoint=endpoint,
                        )

                    logger.warning(
                        "source_request_non_200",
                        endpoint=endpoint,
                        status_code=response.status_code,
                        attempt=attempt,
                    )

                except (
                    httpx.TimeoutException,
                    httpx.ConnectError,
                    httpx.NetworkError,
                ) as e:
                    last_error = e
                    logger.warning(
                        "source_request_network_retry",
                        endpoint=endpoint,
                        attempt=attempt,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

                except Exception as e:
                    last_error = e
                    logger.error(
                        "source_request_unexpected",
                        endpoint=endpoint,
                        attempt=attempt,
                        error=str(e),
                    )
                    break

                # Backoff before retry
                if attempt < self.max_retries:
                    delay = min(
                        self.backoff_base * (2 ** (attempt - 1)), self.backoff_max
                    )
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
            used_endpoint=self.primary_url,
        )
