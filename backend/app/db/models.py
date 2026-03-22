from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, func, ForeignKey, JSON, Float, Integer
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

    session_token: Mapped[str] = mapped_column(String(255), unique=True, nullable=True)

    history: Mapped[list["AnalysisHistory"]] = relationship("AnalysisHistory", back_populates="user")


    def __repr__(self) -> str:
        return f"UserTable(id={self.id}, username='{self.username}')"

class AnalysisHistory(Base):
    __tablename__ = 'analysis_history'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)

    input_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    shap_image_base64: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    user: Mapped["UserTable"] = relationship("UserTable", back_populates="history")