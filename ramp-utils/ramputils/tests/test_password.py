from ramputils.password import hash_password
from ramputils.password import check_password


def test_check_password():
    password = "hjst3789ep;ocikaqjw"
    hashed_password = hash_password(password)
    assert check_password(password, hashed_password)
    assert not check_password("hjst3789ep;ocikaqji", hashed_password)
