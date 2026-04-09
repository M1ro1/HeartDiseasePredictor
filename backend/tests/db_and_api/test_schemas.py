import pytest
from pydantic import ValidationError
from app.db.schemas import UserCreate

def test_user_create_valid():
    user = UserCreate(username="johndoe", email="john@example.com", password="securepassword123")
    assert user.username == "johndoe"

def test_user_create_invalid_email():
    with pytest.raises(ValidationError):
        UserCreate(username="johndoe", email="not-an-email", password="securepassword123")

def test_user_create_short_password():
    with pytest.raises(ValidationError):
        UserCreate(username="johndoe", email="john@example.com", password="short")