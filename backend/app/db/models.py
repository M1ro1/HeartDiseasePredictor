from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, func
from .database import Base
from datetime import datetime

class UserTable(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at:  Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


    def __repr__(self) -> str:
        return f"UserTable(id={self.id}, username='{self.username}')"