from locust import HttpLocust
import os
from locust import TaskSet
from locust import task


class DataboardTasks(TaskSet):
    def on_start(self):
        """ on_start is called when a Locust start before any task
        is scheduled """
        self.login()

    def login(self):
        user_name = os.environ.get("DATABOARD_USERNAME")
        user_pwd = os.environ.get("DATABOARD_PASSWORD")
        self.client.post("/login", {"user_name": user_name,
                                    "password": user_pwd})

    @task(10)
    def index(self):
        self.client.get("/")

    @task(5)
    def about(self):
        self.client.get("/events/iris_test/leaderboard")


class WebsiteUser(HttpLocust):
    host = "http://0.0.0.0:8080"
    task_set = DataboardTasks
    min_wait = 1000  # time in ms between two user actions
    max_wait = 15000  # time in ms between two user actions
