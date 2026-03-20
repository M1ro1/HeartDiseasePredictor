from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from .models import UserTable
from .schemas import UserCreate
from ..security import password_hash

async def get_user_by_username(db: AsyncSession, username: str):
    query = select(UserTable).where(UserTable.username == username)
    result = await db.execute(query)
    return result.scalar_one_or_none()

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