from itsdangerous import URLSafeTimedSerializer

from ramp_frontend import app

ts = URLSafeTimedSerializer(app.config["SECRET_KEY"])
