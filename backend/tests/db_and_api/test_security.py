from app.security import password_hash, verify_password

def test_password_hashing():
    plain_password = "super_secret_password"

    hashed_password = password_hash(plain_password)

    assert plain_password != hashed_password

    assert verify_password(plain_password, hashed_password) is True

    assert verify_password('wrong_password', hashed_password) is False