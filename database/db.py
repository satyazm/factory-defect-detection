"""
SQLite storage for detection events. Swap the SQLALCHEMY_URL for a
PostgreSQL DSN later (e.g. postgresql+psycopg2://...) without touching
the rest of the pipeline — everything else here talks to the ORM layer.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

from sqlalchemy import Boolean, Float, String, DateTime, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

DB_PATH = Path(__file__).resolve().parent / "detections.db"
SQLALCHEMY_URL = f"sqlite:///{DB_PATH}"

_engine = create_engine(SQLALCHEMY_URL, echo=False)


class Base(DeclarativeBase):
    pass


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    camera_id: Mapped[str] = mapped_column(String(64), default="default")
    product_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    class_name: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    bbox_x1: Mapped[float] = mapped_column(Float)
    bbox_y1: Mapped[float] = mapped_column(Float)
    bbox_x2: Mapped[float] = mapped_column(Float)
    bbox_y2: Mapped[float] = mapped_column(Float)
    image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)


class Anomaly(Base):
    __tablename__ = "anomalies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    camera_id: Mapped[str] = mapped_column(String(64), default="default")
    product_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    anomaly_score: Mapped[float] = mapped_column(Float)
    is_anomalous: Mapped[bool] = mapped_column(Boolean, index=True)
    image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)


def init_db() -> None:
    Base.metadata.create_all(_engine)


def log_detection(
    class_name: str,
    confidence: float,
    bbox_xyxy: tuple[float, float, float, float],
    camera_id: str = "default",
    product_id: str | None = None,
    image_path: str | None = None,
) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    with Session(_engine) as session:
        session.add(
            Detection(
                camera_id=camera_id,
                product_id=product_id,
                class_name=class_name,
                confidence=confidence,
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                image_path=image_path,
            )
        )
        session.commit()


def recent_detections(limit: int = 100) -> list[Detection]:
    with Session(_engine) as session:
        stmt = select(Detection).order_by(Detection.timestamp.desc()).limit(limit)
        return list(session.scalars(stmt))


def defect_counts_today() -> dict[str, int]:
    today = dt.datetime.utcnow().date()
    with Session(_engine) as session:
        stmt = select(Detection).where(Detection.timestamp >= dt.datetime.combine(today, dt.time.min))
        rows = session.scalars(stmt).all()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.class_name] = counts.get(row.class_name, 0) + 1
    return counts


def log_anomaly(
    score: float,
    is_anomalous: bool,
    camera_id: str = "default",
    product_id: str | None = None,
    image_path: str | None = None,
) -> None:
    with Session(_engine) as session:
        session.add(
            Anomaly(
                camera_id=camera_id,
                product_id=product_id,
                anomaly_score=score,
                is_anomalous=is_anomalous,
                image_path=image_path,
            )
        )
        session.commit()


def recent_anomalies(limit: int = 100) -> list[Anomaly]:
    with Session(_engine) as session:
        stmt = select(Anomaly).order_by(Anomaly.timestamp.desc()).limit(limit)
        return list(session.scalars(stmt))


def anomaly_stats_today() -> dict:
    today = dt.datetime.utcnow().date()
    with Session(_engine) as session:
        stmt = select(Anomaly).where(Anomaly.timestamp >= dt.datetime.combine(today, dt.time.min))
        rows = session.scalars(stmt).all()
    total = len(rows)
    abnormal = sum(1 for r in rows if r.is_anomalous)
    avg_score = sum(r.anomaly_score for r in rows) / total if total else 0.0
    return {"total": total, "abnormal": abnormal, "avg_score": avg_score}


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
