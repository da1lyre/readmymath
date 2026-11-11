import os
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL environment variable is not set!")

engine = create_async_engine(DATABASE_URL, echo=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Attempt(Base):
    __tablename__ = "attempts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    char_code = Column(String)
    latex_str = Column(String, nullable=True)
    score = Column(Float)
    is_saved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class HighScore(Base):
    __tablename__ = "high_scores"
    user_id = Column(String, primary_key=True)
    score = Column(Integer, default=0)


AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_or_create_user(user_id: str, username: str | None = None, first_name: str | None = None):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.user_id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            user = User(user_id=user_id, username=username, first_name=first_name)
            session.add(user)
        else:
            if username:
                user.username = username
            if first_name:
                user.first_name = first_name

        await session.commit()
        return user


async def get_user_profile(user_id: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.user_id == user_id))
        return result.scalar_one_or_none()


async def save_attempt(user_id: str, char_code: str, latex_str: str, score: float):
    async with AsyncSessionLocal() as session:
        attempt = Attempt(user_id=user_id, char_code=char_code, latex_str=latex_str, score=score)
        session.add(attempt)
        await session.commit()
        return attempt.id


async def save_formula(formula_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Attempt).where(Attempt.id == formula_id))
        attempt = result.scalar_one_or_none()
        if attempt:
            attempt.is_saved = True
            await session.commit()


async def get_saved_formulas(user_id: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Attempt).where(Attempt.user_id == user_id, Attempt.is_saved.is_(True)).order_by(
                Attempt.created_at.desc())
        )
        return result.scalars().all()


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


async def get_char_stats(user_id: str, char_code: str):
    async with AsyncSessionLocal() as session:
        count_res = await session.execute(
            select(func.count(Attempt.id)).where(Attempt.user_id == user_id, Attempt.char_code == char_code)
        )
        count = count_res.scalar() or 0
        avg_res = await session.execute(
            select(func.avg(Attempt.score)).where(Attempt.user_id == user_id, Attempt.char_code == char_code)
        )
        avg = avg_res.scalar()
        return {"attempts": count, "average": round(avg, 1) if avg else 0.0}


async def get_high_score(user_id: str) -> int:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(HighScore).where(HighScore.user_id == user_id))
        hs = result.scalar_one_or_none()
        return hs.score if hs else 0


async def update_high_score(user_id: str, new_score: int) -> int:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(HighScore).where(HighScore.user_id == user_id))
        hs = result.scalar_one_or_none()
        if not hs:
            hs = HighScore(user_id=user_id, score=new_score)
            session.add(hs)
        elif new_score > hs.score:
            hs.score = new_score
        await session.commit()
        return hs.score


async def get_total_attempts(user_id: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(func.count(Attempt.id)).where(Attempt.user_id == user_id)
        )
        return result.scalar() or 0


async def get_leaderboard(limit: int = 10):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User.first_name, User.username, HighScore.score)
            .join(HighScore, User.user_id == HighScore.user_id)
            .where(HighScore.score > 0)
            .order_by(HighScore.score.desc())
            .limit(limit)
        )
        return result.all()


async def get_user_rank(user_id: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(HighScore.score).where(HighScore.user_id == user_id))
        user_score = result.scalar() or 0

        if user_score == 0:
            return None, 0

        rank_res = await session.execute(
            select(func.count(HighScore.user_id)).where(HighScore.score > user_score)
        )
        rank = (rank_res.scalar() or 0) + 1
        return rank, user_score