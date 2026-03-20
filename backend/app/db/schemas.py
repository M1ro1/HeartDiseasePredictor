from pydantic import BaseModel, ConfigDict, Field, EmailStr
from datetime import datetime

class UserCreate(BaseModel):
    username: str = Field(min_length=2, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8, max_length=50)


class UserOut(BaseModel):
    username: str
    email: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)