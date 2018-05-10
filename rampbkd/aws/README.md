Backend support for Amazon EC2 instances.
The goal of this module is to provide a set of
helper functions and CLI command lines 
to train submission(s) on EC2.

In the following is a tutorial showing the full steps starting
from creating the AMI to running submissions on EC2.

# Step 1 : Creating the AMI

In order to create the AMI for a new RAMP challenge, you need to launch an instance on EC2.
Go to https://us-west-2.console.aws.amazon.com/ec2/v2/home, then launch a new instance.
You will need to choose a base AMI. For most cases you will probably just need
the default AMIs provided by amazon such as Ubuntu Server 16.04 LTS.

Once you run the instance, you will need to first install
ramp-workflow (https://github.com/paris-saclay-cds/ramp-workflow).
Then, you should clone the ramp-kits in any folder (we will provide later
the folder in the configuration file), but as a convention we put the 
kit in the folder ~/ramp-kits. Here is an example with the iris
kit:

```
pip install git+https://github.com/paris-saclay-cds/ramp-workflow
mkdir ~/ramp-kits
cd ~/ramp-kits
git clone https://github.com/ramp-kits/iris
```

The second step is to put the private data in the folder data of the kit.
To do that you will have to rsync the private data from the ramp server.
In the ramp server, do this:

```
rsync -avzP /mnt/ramp_data/frontend/ramp-data/iris/data ubuntu@ip:~/ramp-kits/iris
```

where ip is the public ip of the ec2 instance (check the EC2 console to get the IP).

To make sure everything works, try :

```
ramp_test_submission
```


The last step is to install the remaining required packages.

Make sure you install memory_profiler from master if you want to enable memory profiling.

```
pip install git+https://github.com/pythonprofilers/memory_profiler/
```

You can now install all the other packages needed for training the submissions.

Now, you can create the AMI in amazon in the EC2 console.
Go to https://us-west-2.console.aws.amazon.com/ec2/v2/home.
Select the instance, then actions, image, create image.
You can name it according to the ramp kit, e.g., "iris_backend"
to follow the convention. To use the image, you will need to either provide
the image ID, or the image name, as we will se below in Step 2.

# Step 2 : configuration file for ramp-backend

The second step is to prepare a configuration file in the ramp server.
It can be anywhere, by convention it is in /mnt/ramp_data/backend/<event_name>/config.yml
where here event_name is iris. An example is provided in the following:

```

sqlalchemy:
    drivername: postgresql
    username: username
    password: *****
    host: localhost
    port: *****
    database: *****
ramp:
    event_name : iris
aws:
    ami_image_id : ami-0bc19972 OR ami_image_name : iris_backend
    ami_user_name : ubuntu
    instance_type : t2.micro
    key_name: key
    key_path: /home/user/.ssh/id_rsa
    security_group : launch-wizard-1
    remote_ramp_kit_folder : ~/ramp-kits/iris
    local_predictions_folder : ./predictions
    local_log_folder : ./logs
    check_status_interval_secs : 60
    check_finished_training_interval_secs : 60
    train_loop_interval_secs : 60
    memory_profiling : true
    hooks :
        after_sucessful_training: cd /mnt/ramp_data/frontend;fab compute_contributivity:iris;fab update_leaderboards:e=iris
```

The following is an explanation of each field in the aws section.

`event_name` is the event name. This is used by `ramp_aws_train_loop`
(see below) to know the event for which to train new submissions.

`ami_image_id` is the id of the image to use for training the submissions
(the one we created in Step 1). You can get the AMI image ID in the EC2
console, in the tab AMI. It should start with 'ami-'.
The AMI should contain a folder `remote_ramp_kit_folder` (see below)
which contains the ramp kit. In Step 1 we chose `remote_ramp_kit_folder` to be ~/ramp-kits/iris.
Alternatively you can specify the image name rather than the image id, especially if you modify
the image a lot. To do that, you need to use the field `ami_image_name`.

`ami_image_name` is the  name of the image to use for training the submissions.
It is an alternative to `ami_image_id`. That is, you either specify `ami_image_id`
or `ami_image_name`, not both at the same time.

`ami_user_name` is the username to connect with remotely on ec2 instances.

`instance_type` is the instance type (check https://ec2instances.info/).

`key_name` is the name of the key to connect with, so `key_name` should
exist im amazon (check key pairs in EC2 console). 

`security_group` is the name of the security group to use.
Security groups control which ports are accepted/blocked inbound or outbound.
They can be created in the web app of amazon. Use `default`
to use the default one or choose one from the EC2 console, in the tab
security group.

`remote_ramp_kit_folder` is the folder in the ec2 instance
where the ramp-kit will reside. In Step 1 we chose to put it 
in ~/ramp-kits/iris. It should be possible to launch 
`ramp_test_submission` with success in that folder.

`local_predictions_folder` is the local folder where the predictions are
downloaded (from the ec2 instance).

`local_log_folder` is the local folder where the logs are downloaded
(from the ec2 instance). The logs contain the standard output and error 
obtained from running `ramp_test_submission` for a given submission.

`check_status_interval_secs` is the number of secs to wait until we
recheck whether an ec2 instance is ready to be used.

`check_finished_training_interval_secs` is the number of secs to wait
until we recheck whether the training of a submission in an ec2
instance is finished.

`train_loop_interval_secs` is the number of secs to wait each time we
process new events in `train_loop`

`memory_profiling` turns on (or off) memory profiling to know how much memory was
needed by a submission

`hooks` is for specifying local commands that will run for after some event such as when
a submission has been trained successfully. Hooks available are:

### hooks

after_sucessful_training: `command`. it runs the given command each time a submission is 
successfully trained.

# Step 3: Using the CLI

Two command line interfaces are provided, ramp_aws_train and
ramp_aws_train_loop.

## ramp_aws_train

To train a single submission on aws, you can use ramp_aws_train.
To train a submission, use the following:

```
ramp_aws_train config.yml --event=<event name> --team=<team name> --name=<submission name>
```

By default a new ec2 instance will be created then training will be done there, 
then the instance will be killed after training.

If you want to train on an existing (running) instance just add the option
--instance-id like the following:

```
ramp_aws_train config.yml --event=<event name> --team=<team name> --name=<submission name>  --instance-id=<instance id>
```

To find the instance id, you have to check the EC2 console.


### ramp_aws_train_loop

To launch a training loop that will automatically listen for new submissions and run them, use the following:

```
ramp_aws_train_loop config.yml
```
