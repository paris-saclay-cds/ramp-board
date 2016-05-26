# Defining Locust tests. Run it with: locust -f locustfile.py and
# go to http://127.0.0.1:8089/
import os
import numpy as np
from locust import HttpLocust
from locust import TaskSet
from locust import task
import databoard.config as config


event_name = 'iris_test'  # name of the event to be tested
databoard_path = '/tmp/databoard_test'  # path where databoard is published

assert not config.WTF_CSRF_ENABLED, 'Set WTF_CSRF_ENABLED to False'


class DataboardTasks(TaskSet):
    """
    Simulates the behaviour of a user who is already registered
    and has already signed up for the event
    """
    def on_start(self):
        """ on_start is called when a Locust start before any task
        is scheduled """
        self.login()

    def login(self):
        # response = self.client.get("/login")
        # csrf_token = response.cookies["session"]
        user_name = os.environ.get("DATABOARD_USERNAME")
        user_pwd = os.environ.get("DATABOARD_PASSWORD")
        self.client.post("/login", {"user_name": user_name,
                                    "password": user_pwd})  # ,
        #                            "csrf_token": csrf_token},
        #                 headers={"X-CSRFToken": csrf_token})

    @task(10)
    def leaderboard(self):
        self.client.get("/events/" + event_name + "/leaderboard")

    @task(5)
    def submit(self):
        submission_id = np.random.randint(1000000)
        submission_name = "locust_%i" % submission_id
        print(submission_name)
        self.client.post("/events/" + event_name + "/sandbox",
                         {'submission_name': submission_name})
        os.system('cd %s; fab train_test:%s,t=%s,s=%s'
                  % (databoard_path, event_name,
                     os.environ.get("DATABOARD_USERNAME"),
                     submission_name))


class WebsiteUser(HttpLocust):
    host = "http://0.0.0.0:8080"
    task_set = DataboardTasks
    min_wait = 1000  # time in ms between two user actions
    max_wait = 15000  # time in ms between two user actions
