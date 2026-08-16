from chatpilot.apps.web.models.auths import Auths
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def signup(email: str, password: str, name: str, role: str = 'student') -> dict:
    hashed = pwd_context.hash(password)

    user = Auths.insert_new_auth(email.lower(), hashed, name, role)

signup(
    email='test@hku.hk',
    password='test',
    name='Troy'
) # Test Successful