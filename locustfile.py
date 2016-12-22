# Defining Locust tests. Run it with: locust -f locustfile.py and
# go to http://127.0.0.1:8089/
import os
import numpy as np
from locust import HttpLocust
from locust import TaskSet
from locust import task


event_name = 'el_nino'  # name of the event to be tested


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
        r = self.client.post("/login", {"user_name": user_name,
                                        "password": user_pwd})  # ,
        #                            "csrf_token": csrf_token},
        #                 headers={"X-CSRFToken": csrf_token})
        print('LOGIN - ' + str(r.status_code))

    @task(10)
    def user(self):
        self.client.get("/user")

    @task(10)
    def leaderboard(self):
        self.client.get("/events/" + event_name + "/leaderboard")

    @task(10)
    def leaderboard_drug_spectra(self):
        self.client.get("/events/drug_spectra/leaderboard")

    @task(10)
    def submissions_drug_spectra(self):
        self.client.get("/events/drug_spectra/my_submissions")

    @task(10)
    def submissions_el_nino(self):
        self.client.get("/events/el_nino/my_submissions")

    @task(10)
    def description_drug_spectra(self):
        self.client.get("/events/drug_spectra")

    @task(10)
    def description_el_nino(self):
        self.client.get("/events/el_nino")

    # @task(5)
    # def submit(self):
    #     submission_id = np.random.randint(1000000)
    #     submission_name = "locust_%i" % submission_id
    #     r = self.client.post("/events/" + event_name + "/sandbox",
    #                          {'submission_name': submission_name})
    #     print('SUBMISSION - ' + str(r.status_code))


class WebsiteUser(HttpLocust):
    # host = "http://0.0.0.0:8080"
    host = "http://134.158.75.211"
    task_set = DataboardTasks
    min_wait = 1000  # time in ms between two user actions
    max_wait = 15000  # time in ms between two user actions
