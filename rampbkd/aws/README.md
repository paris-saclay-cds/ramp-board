Backend support for Amazon EC2 instances.
The goal of this module is to provide a set of
helper functions to train a submission on an ec2 instance.
The first step is to add a section 'aws' on the configuration
file as the following.

## Configuration details

```
aws:
    ami_image_id : ami-0bc19972
    ami_user_name : ubuntu
    instance_type : t2.micro
    key_name: key
    key_path: /home/user/.ssh/id_rsa
    security_group : launch-wizard-1
    remote_ramp_kit_folder : ~/ramp/iris
    local_predictions_folder : ./predictions
    local_log_folder : ./logs
    check_status_interval_secs : 60
    check_finished_training_interval_secs : 60
    train_loop_interval_secs : 60
    memory_profiling : true
```
 
`ami_image_id` is the id of the image to use, it should start with 'ami-'.
The AMI should contain a folder `remote_ramp_kit_folder` (see below)
which contains the ramp kit.

`ami_user_name` is the username to connect with remotely on ec2 instances.

`instance_type` is the instance type (check https://ec2instances.info/).

`key_name` is the name of the key to connect with, so `key_name` should
exist im amazon. It can be created using their web app, or manually via
`aws` like this :
```aws ec2 import-key-pair --key-name <put key name here> --public-key-material "<put public key here>"```

`security_group` is the name of the security group to use.
Security groups control which ports are accepted/blocked inbound or outbound.
They can be created in the web app of amazon. Use `default`
to use the default one.

`remote_ramp_kit_folder` is the folder in the ec2 instance
where the ramp-kit will reside. It should
be possible to launch `ramp_test_submission` in that folder.

`local_predictions_folder` is the local folder where the predictions are
downloaded (from the ec2 instance).

`local_log_folder` is the local folder where the logs are downloaded
(from the ec2 instance). The logs contain the standard output obtained
from running `ramp_test_submission` for a given submission.

`check_status_interval_secs` is the number of secs to wait until we
recheck whether an ec2 instance is ready to be used.

`check_finished_training_interval_secs` is the number of secs to wait
until we recheck whether the training of a submission in an ec2
instance is finished.

`train_loop_interval_secs` is the number of secs to wait each time we
process new events in `train_loop`

`memory_profiling` turns on memory profiling to know how much memory was
needed by a submission

## Using the API
Once configuration is ready, the most straighforward way to use the API is
to use the function `launch_ec2_instance_and_train`. It does the full pipeline
in one pass. That is, it launches an ec2 instance, waits until it is
ready, upload submission, starts training, wait for training to
finish, download the predictions and logs, store the predictions on the
database, then terminate the ec2 instance.

 ```python
from rampbkd.config import read_backend_config
conf = read_backend_config('config.yml')
launch_ec2_instance_and_train(conf, submission_id)
 ```
 
Another way to use the API is to run a loop that listens for new
submissions and  run them.

```python
from rampbkd.config import read_backend_config
conf = read_backend_config('config.yml')
train_loop(conf)
```

The other available functions can be used to do something more custom.
One could imagine to launch a pool of ec2 instances first, then have
a training loop which waits for submissions and run them in an ec2 instance
(nothing prevents us to train multiple submissions in the same ec2 instance).
The pool size could be adapted automatically to the need. Also, different
submissions could need different types of machines (GPU vs CPU).

## Using the CLI

Two command line interfaces are provided.
To train a single submission on aws, you can use:

```
ramp_aws_train config.yml <submission_id>
```

To launch the training loop, you can use:

```
ramp_aws_train_loop config.yml
```
