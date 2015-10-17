fab publish_test:mortality_prediction_local_test
cd /tmp/databoard_local/databoard_mortality_prediction_8080
fab test_ramp

fab publish_test:iris_local_test
cd /tmp/databoard_local/databoard_iris_8080
fab test_ramp

