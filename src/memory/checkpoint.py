"""LangGraph-compatible PostgreSQL checkpoint saver.

Implements ``BaseCheckpointSaver`` from LangGraph >= 0.2 so that agent
graphs can persist their full state between invocations.  Only the async
methods are implemented; the synchronous counterparts raise
``NotImplementedError`` because agents in this application always use
``ainvoke``.

Schema
------
``langgraph_checkpoints``
    One row per checkpoint, keyed by (thread_id, checkpoint_ns, checkpoint_id).

``langgraph_checkpoint_writes``
    In-progress writes (pending channel updates) associated with a
    checkpoint.  Populated via ``aput_writes`` and used by LangGraph to
    resume interrupted executions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional

import sqlalchemy as sa
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.core.logging import get_logger, log_debug
from src.models.common import utcnow

logger: logging.Logger = get_logger(__name__)

_metadata = sa.MetaData()

_checkpoints = sa.Table(
    "langgraph_checkpoints",
    _metadata,
    sa.Column("thread_id", sa.String(128), nullable=False),
    sa.Column("checkpoint_ns", sa.String(128), nullable=False, server_default=""),
    sa.Column("checkpoint_id", sa.String(128), nullable=False),
    sa.Column("parent_checkpoint_id", sa.String(128), nullable=True),
    sa.Column("checkpoint", sa.JSON, nullable=False),
    sa.Column("metadata", sa.JSON, nullable=False, server_default="{}"),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint("thread_id", "checkpoint_ns", "checkpoint_id"),
)

_checkpoint_writes = sa.Table(
    "langgraph_checkpoint_writes",
    _metadata,
    sa.Column("thread_id", sa.String(128), nullable=False),
    sa.Column("checkpoint_ns", sa.String(128), nullable=False, server_default=""),
    sa.Column("checkpoint_id", sa.String(128), nullable=False),
    sa.Column("task_id", sa.String(128), nullable=False),
    sa.Column("idx", sa.Integer, nullable=False),
    sa.Column("channel", sa.Text, nullable=False),
    sa.Column("type", sa.Text, nullable=True),
    sa.Column("blob", sa.LargeBinary, nullable=True),
    sa.PrimaryKeyConstraint("thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"),
)


def _extract_configurable(config: RunnableConfig) -> tuple[str | None, str, str | None]:
    """Return (thread_id, checkpoint_ns, checkpoint_id) from a RunnableConfig."""
    cfg = config.get("configurable") or {}
    return cfg.get("thread_id"), cfg.get("checkpoint_ns", ""), cfg.get("checkpoint_id")


class PostgresCheckpointSaver(BaseCheckpointSaver):  # type: ignore[misc]
    """Async-only LangGraph checkpoint saver backed by PostgreSQL.

    All synchronous methods raise ``NotImplementedError``; always use
    ``ainvoke`` / ``astream`` with the agent graph.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__()
        self._engine = engine

    @classmethod
    def from_url(cls, url: str) -> "PostgresCheckpointSaver":
        """Create a saver from a SQLAlchemy async database URL."""
        return cls(create_async_engine(url))

    async def create_tables(self) -> None:
        """Create checkpoint tables if they do not already exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all, checkfirst=True)

    # ------------------------------------------------------------------
    # Sync abstract methods — not supported (raise to catch misuse)
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError(
            "PostgresCheckpointSaver is async-only; use aget_tuple"
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> RunnableConfig:
        raise NotImplementedError("PostgresCheckpointSaver is async-only; use aput")

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        raise NotImplementedError("PostgresCheckpointSaver is async-only; use alist")

    # ------------------------------------------------------------------
    # Async implementations
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id, checkpoint_ns, checkpoint_id = _extract_configurable(config)
        if not thread_id:
            return None

        stmt = sa.select(_checkpoints).where(
            _checkpoints.c.thread_id == thread_id,
            _checkpoints.c.checkpoint_ns == checkpoint_ns,
        )
        if checkpoint_id is not None:
            stmt = stmt.where(_checkpoints.c.checkpoint_id == checkpoint_id)
        else:
            stmt = stmt.order_by(_checkpoints.c.created_at.desc()).limit(1)

        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).mappings().first()

        if row is None:
            return None

        checkpoint_data: Checkpoint = self.serde.loads(
            json.dumps(row["checkpoint"]).encode()
        )
        parent_config: Optional[RunnableConfig] = None
        if row["parent_checkpoint_id"]:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": str(row["parent_checkpoint_id"]),
                }
            }

        log_debug(
            logger,
            "memory.checkpoint.loaded",
            "Checkpoint loaded",
            thread_id=thread_id,
            checkpoint_id=str(row["checkpoint_id"]),
        )

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": str(row["checkpoint_id"]),
                }
            },
            checkpoint=checkpoint_data,
            metadata=row["metadata"] or {},
            parent_config=parent_config,
            pending_writes=None,
        )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> RunnableConfig:
        thread_id, checkpoint_ns, parent_checkpoint_id = _extract_configurable(config)
        thread_id = thread_id or ""
        checkpoint_id: str = checkpoint["id"]
        checkpoint_data = json.loads(self.serde.dumps(checkpoint))

        async with self._engine.begin() as conn:
            exists = (
                await conn.execute(
                    sa.select(sa.func.count())
                    .select_from(_checkpoints)
                    .where(
                        _checkpoints.c.thread_id == thread_id,
                        _checkpoints.c.checkpoint_ns == checkpoint_ns,
                        _checkpoints.c.checkpoint_id == checkpoint_id,
                    )
                )
            ).scalar() or 0

            if exists:
                await conn.execute(
                    sa.update(_checkpoints)
                    .where(
                        _checkpoints.c.thread_id == thread_id,
                        _checkpoints.c.checkpoint_ns == checkpoint_ns,
                        _checkpoints.c.checkpoint_id == checkpoint_id,
                    )
                    .values(checkpoint=checkpoint_data, metadata=dict(metadata))
                )
            else:
                await conn.execute(
                    sa.insert(_checkpoints).values(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        parent_checkpoint_id=parent_checkpoint_id,
                        checkpoint=checkpoint_data,
                        metadata=dict(metadata),
                        created_at=utcnow(),
                    )
                )

        log_debug(
            logger,
            "memory.checkpoint.saved",
            "Checkpoint saved",
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def alist(  # type: ignore[override]
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        thread_id = ((config or {}).get("configurable") or {}).get("thread_id")

        stmt = sa.select(_checkpoints).order_by(_checkpoints.c.created_at.desc())
        if thread_id:
            stmt = stmt.where(_checkpoints.c.thread_id == thread_id)
        if before is not None:
            before_id = ((before.get("configurable")) or {}).get("checkpoint_id")
            if before_id:
                stmt = stmt.where(_checkpoints.c.checkpoint_id < before_id)
        if limit is not None:
            stmt = stmt.limit(limit)

        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()

        for row in rows:
            checkpoint_data: Checkpoint = self.serde.loads(
                json.dumps(row["checkpoint"]).encode()
            )
            parent_config: Optional[RunnableConfig] = None
            if row["parent_checkpoint_id"]:
                parent_config = {
                    "configurable": {
                        "thread_id": str(row["thread_id"]),
                        "checkpoint_ns": str(row["checkpoint_ns"]),
                        "checkpoint_id": str(row["parent_checkpoint_id"]),
                    }
                }
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": str(row["thread_id"]),
                        "checkpoint_ns": str(row["checkpoint_ns"]),
                        "checkpoint_id": str(row["checkpoint_id"]),
                    }
                },
                checkpoint=checkpoint_data,
                metadata=row["metadata"] or {},
                parent_config=parent_config,
                pending_writes=None,
            )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        if not writes:
            return

        thread_id, checkpoint_ns, checkpoint_id = _extract_configurable(config)
        thread_id = thread_id or ""
        checkpoint_id = checkpoint_id or ""

        async with self._engine.begin() as conn:
            for idx, (channel, value) in enumerate(writes):
                try:
                    type_str, blob = self.serde.dumps_typed(value)
                except AttributeError:
                    type_str = "json"
                    blob = self.serde.dumps(value)

                exists = (
                    await conn.execute(
                        sa.select(sa.func.count())
                        .select_from(_checkpoint_writes)
                        .where(
                            _checkpoint_writes.c.thread_id == thread_id,
                            _checkpoint_writes.c.checkpoint_ns == checkpoint_ns,
                            _checkpoint_writes.c.checkpoint_id == checkpoint_id,
                            _checkpoint_writes.c.task_id == task_id,
                            _checkpoint_writes.c.idx == idx,
                        )
                    )
                ).scalar() or 0

                if exists:
                    await conn.execute(
                        sa.update(_checkpoint_writes)
                        .where(
                            _checkpoint_writes.c.thread_id == thread_id,
                            _checkpoint_writes.c.checkpoint_ns == checkpoint_ns,
                            _checkpoint_writes.c.checkpoint_id == checkpoint_id,
                            _checkpoint_writes.c.task_id == task_id,
                            _checkpoint_writes.c.idx == idx,
                        )
                        .values(channel=channel, type=type_str, blob=blob)
                    )
                else:
                    await conn.execute(
                        sa.insert(_checkpoint_writes).values(
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            checkpoint_id=checkpoint_id,
                            task_id=task_id,
                            idx=idx,
                            channel=channel,
                            type=type_str,
                            blob=blob,
                        )
                    )


def make_memory_saver(url: str) -> PostgresCheckpointSaver:
    """Create a ``PostgresCheckpointSaver`` from a database URL.

    Convenience factory for dependency injection — callers supply the URL
    from ``get_settings().postgres_url`` and receive a ready-to-use saver.
    ``create_tables()`` must be called before the first ``ainvoke``.
    """
    return PostgresCheckpointSaver.from_url(url)
