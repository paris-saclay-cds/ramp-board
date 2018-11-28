from itsdangerous import URLSafeTimedSerializer

from databoard import app

ts = URLSafeTimedSerializer(app.config["SECRET_KEY"])
