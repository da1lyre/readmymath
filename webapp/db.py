import os
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, select, func

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL environment variable is not set!")

engine = create_async_engine(DATABASE_URL, echo=False)
Base = declarative_base()

class Attempt(Base):
    __tablename__ = "attempts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    char_code = Column(String)
    score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def save_attempt(user_id: str, char_code: str, score: float):
    async with AsyncSessionLocal() as session:
        attempt = Attempt(user_id=user_id, char_code=char_code, score=score)
        session.add(attempt)
        await session.commit()

async def get_history(user_id: str, limit: int = 10):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Attempt)
            .where(Attempt.user_id == user_id)
            .order_by(Attempt.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

async def get_average_score(user_id: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(func.avg(Attempt.score)).where(Attempt.user_id == user_id)
        )
        avg = result.scalar()
        return round(avg, 1) if avg else 0.0