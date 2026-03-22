from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from .db.models import UserTable
from fastapi import Header, HTTPException, Depends
from sqlalchemy.future import select
from .db.database import get_db

import uuid

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def password_hash(password:str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password:str, hashed_password:str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


async def create_user_session(db: AsyncSession, user: UserTable):
    token = str(uuid.uuid4())
    user.session_token = token
    await db.commit()
    return token

async def get_current_user(
        db: AsyncSession = Depends(get_db),
        x_token: str = Header(None)
):
    if not x_token:
        return None

    result = await db.execute(select(UserTable).where(UserTable.session_token == x_token))
    user = result.scalar_one_or_none()
    return user