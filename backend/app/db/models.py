from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String

from .database import Base

class UserTable(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)

    def __repr__(self) -> str:
        return f"UserTable(id={self.id}, username='{self.username}')"