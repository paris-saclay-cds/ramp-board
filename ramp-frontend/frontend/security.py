from itsdangerous import URLSafeTimedSerializer

from frontend import app

ts = URLSafeTimedSerializer(app.config["SECRET_KEY"])
