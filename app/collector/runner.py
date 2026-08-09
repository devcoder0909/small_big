"""Collector runner — main 24/7 collection loop."""

import asyncio
import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.database import async_session_factory
from app.collector.client import SourceClient
from app.collector.parser import (
    parse_history_response,
    compute_payload_hash,
    extract_service_time,
)
from app.collector.validator import validate_batch
from app.collector.deduplicator import upsert_batch, get_total_record_count
from app.models.source_request import SourceRequest
from app.models.raw_response import RawResponse
from app.models.system_heartbeat import SystemHeartbeat
from app.models.data_quality import DataQualityEvent

logger = get_logger(__name__)


class CollectorRunner:
    """Main collector orchestrator — runs 24/7 fetch-parse-validate-persist loop."""

    def __init__(self):
        self.settings = get_settings()
        self.client = SourceClient()
        self.start_time = time.monotonic()
        self.total_new_records = 0
        self.total_duplicates = 0
        self.total_errors = 0
        self.last_known_issue_id: str | None = None

    async def _save_source_request(
        self, session: AsyncSession, fetch_result
    ) -> int:
        """Persist source request audit log."""
        sr = SourceRequest(
            requested_at=fetch_result.requested_at,
            request_timestamp_ms=fetch_result.request_timestamp_ms,
            http_status=fetch_result.status_code,
            response_time_ms=fetch_result.response_time_ms,
            success=fetch_result.success,
            error_type=fetch_result.error_type,
            error_message=fetch_result.error_message,
            records_received=None,
        )
        session.add(sr)
        await session.flush()
        return sr.id

    async def _save_raw_response(
        self, session: AsyncSession, source_request_id: int, payload: dict
    ) -> int:
        """Persist raw JSON response."""
        rr = RawResponse(
            source_request_id=source_request_id,
            received_at=datetime.now(timezone.utc),
            payload=payload,
            payload_hash=compute_payload_hash(payload),
        )
        session.add(rr)
        await session.flush()
        return rr.id

    async def _update_heartbeat(
        self, session: AsyncSession, status: str, new_record: bool = False
    ):
        """Update system heartbeat."""
        now = datetime.now(timezone.utc)
        uptime = int(time.monotonic() - self.start_time)
        bind = session.get_bind()
        dialect_name = bind.dialect.name if bind else "postgresql"

        if dialect_name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = pg_insert(SystemHeartbeat).values(
                service_name="collector",
                last_heartbeat=now,
                last_successful_fetch=now if status in ("HEALTHY",) else None,
                last_new_record=now if new_record else None,
                status=status,
                total_records=self.total_new_records,
                total_errors=self.total_errors,
                total_duplicates=self.total_duplicates,
                uptime_seconds=uptime,
                created_at=now,
                updated_at=now,
            )
            update_dict = {
                "last_heartbeat": now,
                "status": status,
                "total_records": self.total_new_records,
                "total_errors": self.total_errors,
                "total_duplicates": self.total_duplicates,
                "uptime_seconds": uptime,
                "updated_at": now,
            }
            if status == "HEALTHY":
                update_dict["last_successful_fetch"] = now
            if new_record:
                update_dict["last_new_record"] = now

            stmt = stmt.on_conflict_do_update(
                index_elements=["service_name"],
                set_=update_dict,
            )
            await session.execute(stmt)
        else:
            # Fallback for SQLite / test databases
            result = await session.execute(
                select(SystemHeartbeat).where(SystemHeartbeat.service_name == "collector")
            )
            hb = result.scalar_one_or_none()

            if hb is None:
                hb = SystemHeartbeat(
                    service_name="collector",
                    last_heartbeat=now,
                    last_successful_fetch=now if status in ("HEALTHY",) else None,
                    last_new_record=now if new_record else None,
                    status=status,
                    total_records=self.total_new_records,
                    total_errors=self.total_errors,
                    total_duplicates=self.total_duplicates,
                    uptime_seconds=uptime,
                    created_at=now,
                    updated_at=now,
                )
                session.add(hb)
            else:
                hb.last_heartbeat = now
                hb.status = status
                hb.total_records = self.total_new_records
                hb.total_errors = self.total_errors
                hb.total_duplicates = self.total_duplicates
                hb.uptime_seconds = uptime
                hb.updated_at = now
                if status == "HEALTHY":
                    hb.last_successful_fetch = now
                if new_record:
                    hb.last_new_record = now

    async def _record_data_quality_event(
        self,
        session: AsyncSession,
        event_type: str,
        severity: str,
        description: str,
        issue_id: str | None = None,
        details: str | None = None,
    ):
        """Record a data quality event."""
        event = DataQualityEvent(
            event_type=event_type,
            severity=severity,
            issue_id=issue_id,
            description=description,
            details=details,
        )
        session.add(event)

    async def _detect_missing_issues(
        self, session: AsyncSession, parsed_results: list
    ):
        """Detect gaps in sequential issue IDs."""
        if len(parsed_results) < 2:
            return

        ids = sorted([r.issue_id for r in parsed_results])

        has_gap = False
        for i in range(1, len(ids)):
            try:
                prev_num = int(ids[i - 1])
                curr_num = int(ids[i])
                gap = curr_num - prev_num
                if gap > 1:
                    has_gap = True
                    for missing in range(prev_num + 1, curr_num):
                        logger.warning(
                            "missing_issue_detected",
                            missing_id=str(missing),
                            between=(ids[i - 1], ids[i]),
                        )
                        await self._record_data_quality_event(
                            session,
                            event_type="missing_issue",
                            severity="WARNING",
                            description=f"Missing issue ID: {missing}",
                            issue_id=str(missing),
                            details=f"Gap between {ids[i-1]} and {ids[i]}",
                        )
            except ValueError:
                pass

        if has_gap:
            try:
                from app.services.recovery_service import recover_missing_records
                rec_res = await recover_missing_records(session)
                logger.info("inline_gap_recovery_completed", result=rec_res)
            except Exception as rec_err:
                logger.warning("inline_gap_recovery_failed", error=str(rec_err))

    async def run_single_cycle(self) -> dict:
        """
        Execute a single collection cycle:
        fetch → parse → validate → persist → detect anomalies → heartbeat
        """
        cycle_result = {
            "success": False,
            "new_records": 0,
            "duplicates": 0,
            "errors": 0,
        }

        _trigger_issue_id = None  # Captured inside transaction, used after commit

        try:
            fetch_result = await self.client.fetch_history()

            async with async_session_factory() as session:
                async with session.begin():
                    sr_id = await self._save_source_request(session, fetch_result)

                    if not fetch_result.success or not fetch_result.data:
                        self.total_errors += 1
                        await self._update_heartbeat(session, "DEGRADED")
                        await self._record_data_quality_event(
                            session,
                            event_type="api_failure",
                            severity="ERROR",
                            description=f"API request failed: {fetch_result.error_type}",
                            details=fetch_result.error_message,
                        )
                        cycle_result["errors"] = 1
                        return cycle_result

                    rr_id = await self._save_raw_response(
                        session, sr_id, fetch_result.data
                    )

                    raw_list = fetch_result.data.get("data", {}).get("list", [])
                    sr = await session.get(SourceRequest, sr_id)
                    if sr:
                        sr.records_received = len(raw_list)

                    try:
                        parsed_results = parse_history_response(fetch_result.data)
                    except ValueError as e:
                        logger.error("parse_failure", error=str(e))
                        self.total_errors += 1
                        await self._record_data_quality_event(
                            session,
                            event_type="parse_failure",
                            severity="ERROR",
                            description=f"Parse error: {e}",
                        )
                        await self._update_heartbeat(session, "DEGRADED")
                        cycle_result["errors"] = 1
                        return cycle_result

                    if not parsed_results:
                        await self._update_heartbeat(session, "HEALTHY")
                        cycle_result["success"] = True
                        return cycle_result

                    valid_results, validation_errors = validate_batch(parsed_results)

                    for verr in validation_errors:
                        await self._record_data_quality_event(
                            session,
                            event_type="invalid_result",
                            severity="WARNING",
                            description=f"Validation failed: {verr['errors']}",
                            issue_id=verr.get("issue_id"),
                        )

                    await self._detect_missing_issues(session, valid_results)

                    source_time = extract_service_time(fetch_result.data)
                    batch_result = await upsert_batch(
                        session,
                        valid_results,
                        source_url=self.settings.source_url,
                        raw_response_id=rr_id,
                        source_created_at=source_time,
                    )

                    self.total_new_records += batch_result["new_records"]
                    self.total_duplicates += batch_result["duplicates"]
                    self.total_errors += batch_result["errors"]

                    has_new = batch_result["new_records"] > 0
                    if has_new:
                        from app.services.cache_service import cache
                        cache.clear()

                    await self._update_heartbeat(
                        session, "HEALTHY", new_record=has_new
                    )

                    if valid_results:
                        self.last_known_issue_id = valid_results[0].issue_id

                    cycle_result.update({
                        "success": True,
                        "new_records": batch_result["new_records"],
                        "duplicates": batch_result["duplicates"],
                        "errors": batch_result["errors"],
                    })

                    if has_new:
                        # Capture latest issue ID for pipeline trigger after commit
                        _trigger_issue_id = valid_results[0].issue_id if valid_results else None
                        logger.info(
                            "cycle_complete",
                            new_records=batch_result["new_records"],
                            duplicates=batch_result["duplicates"],
                            latest_issue=_trigger_issue_id,
                        )
                    else:
                        logger.debug("no_new_data")

                # Transaction committed — trigger prediction pipeline immediately
                if _trigger_issue_id:
                    try:
                        from app.services.prediction_pipeline import pipeline
                        asyncio.create_task(pipeline.trigger_new_result(_trigger_issue_id))
                    except Exception as pipe_err:
                        logger.warning("pipeline_trigger_error", error=str(pipe_err))

        except Exception as e:
            logger.error("cycle_error", error=str(e), error_type=type(e).__name__)
            self.total_errors += 1
            cycle_result["errors"] = 1

            try:
                async with async_session_factory() as session:
                    async with session.begin():
                        await self._update_heartbeat(session, "ERROR")
            except Exception:
                pass

        return cycle_result

    async def run_forever(self):
        """Run the collector loop forever, never terminating on recoverable errors."""
        setup_logging()
        logger.info(
            "collector_starting",
            poll_interval=self.settings.poll_interval_seconds,
            source_url=self.settings.source_url,
        )

        # Execute initial startup recovery to fetch any missing records during downtime
        try:
            from app.services.recovery_service import recover_missing_records
            async with async_session_factory() as session:
                async with session.begin():
                    rec_result = await recover_missing_records(session)
                    logger.info("startup_recovery_completed", result=rec_result)
        except Exception as rec_err:
            logger.warning("startup_recovery_failed", error=str(rec_err))

        while True:
            try:
                await self.run_single_cycle()
            except Exception as e:
                logger.error(
                    "collector_loop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )

            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("collector_shutting_down")
        await self.client.close()


async def main():
    """Entry point for the collector process."""
    runner = CollectorRunner()
    try:
        await runner.run_forever()
    except KeyboardInterrupt:
        logger.info("collector_interrupted")
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
