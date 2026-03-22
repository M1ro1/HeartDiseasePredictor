from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from .models import UserTable
from .schemas import UserCreate
from .database import get_db
from ..security import password_hash, verify_password
from fastapi import Depends, Header
import uuid

async def create_user_session(db: AsyncSession, user: UserTable):
    token = str(uuid.uuid4())
    user.session_token = token
    await db.commit()
    return token

async def get_user_by_username(db: AsyncSession, username: str):
    query = select(UserTable).where(UserTable.username == username)
    result = await db.execute(query)
    return result.scalar_one_or_none()

async def get_current_user(
        db: AsyncSession = Depends(get_db),
        x_token: str = Header(None)
):
    if not x_token:
        return None

    result = await db.execute(select(UserTable).where(UserTable.session_token == x_token))
    user = result.scalar_one_or_none()
    return user

async def create_user(db: AsyncSession, user_data: UserCreate):

    hashed_pass = password_hash(user_data.password)

    new_user = UserTable(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_pass,
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

async def login_user(db: AsyncSession, username: str, password_attempt: str):
    user_info = await get_user_by_username(db, username)

    if user_info is None or not (verify_password(password_attempt, user_info.hashed_password)):
        return False

    return user_info