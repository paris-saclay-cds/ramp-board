.. _all_workers:

Notes regarding the RAMP workers
================================

The RAMP "worker" is the object responsible of actually running (training and
evaluating) a submission and returning back the results (predictions and
scores).
The workers are typically launched from a loop set up for a specific event on
the server. The worker can then run the submission either locally on the server
or on a remote machine.

The type of worker used in a certain event is specified in the ``worker``
section of the yml configuration file, using the ``worker_type`` key (see
:ref:`deploy-ramp-event`). Additional keys can be required depending on
the worker type.

Available workers:

* The :class:`ramp_engine.local.CondaEnvWorker` will run the submission in
  a specific isolated conda environment. This worker is specified as
  ``worker_type: conda`` and requires an additional key ``conda_env``
  specifying the name of the conda environment.

* The :class:`ramp_engine.aws.AWSWorker` will send the submission to an AWS
  instance and copy back the results. This worker is specified as
  ``worker_type: aws``, and for more details on the setup and configuration,
  see below.

.. _conda_env_worker:

Running submissions in a Conda environment
------------------------------------------

As previously mentioned, the :class:`ramp_engine.local.CondaEnvWorker` workers
rely on a Conda environment to execute the submission.

For instance, if you would like to run the `Iris challenge
<https://github.com/ramp-kits/iris>`_, the starting kit requires numpy, pandas,
and scikit-learn. Thus, the environment used to run the initial submission
should have these libraries installed, in addition to ramp-workflow.

Subsequently, you can create such environment and use it in the event
configuration::

      ~ $ conda create --name ramp-iris pip numpy pandas scikit-learn
      ~ $ conda activate ramp-iris
      ~ $ pip install ramp-workflow

If you are using a ramp-kit from the `Paris-Saclay CDS
<https://github.com/ramp-kits>`_, each kit will provide either an
``environment.yml`` or ``ami_environment.yml`` file which you can use to create
the environment::

      conda create --name ramp-iris --file environment.yml

Alternatively, you can include an `environment.yml` file inside the `ramp-kit`
directory and use the following to install the environment::

      ~/ramp_deployment $ ramp setup create-conda-env --event-config events/iris_test/config.yml

You can update an existing environment with the following::

      ~/ramp_deployment $ ramp setup update-conda-env --event-config events/iris_test/config.yml

Running submissions on Amazon Web Services (AWS)
------------------------------------------------

The AWS worker will train and evaluate the submissions on an AWS instance,
which can be useful if the server itself has not much resources or if you need
specific resources (e.g. GPU's).

Summary of steps:

1. Launch AWS instance and connect to it.
2. Prepare the instance for running submissions.
3. Save the instance as an Amazon Machine Image (AMI).
4. Update the event ``config.yml`` file on the RAMP server.

.. _launch_aws:

Launching AWS instance
^^^^^^^^^^^^^^^^^^^^^^

Follow `AWS getting started
<https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html>`_
to launch an Amazon EC2 Linux instance. Amazon can create a new key-pair
for your instance, which you can download. You need to store this in a
hidden directory (e.g., ``.ssh/``) and change the rights to file permissions
to only read for owner. You can use::

      ~ $ chmod 400 <path_to_aws_key>

To connect to your instance via ssh, use::

      ~ $ ssh -i /path/my-key-pair.pem my-instance-user-name@my-instance-public-dns-name

`'my-instance-user-name'` depends on the type of instance you picked but is
commonly 'ec2-user' or 'ubuntu'. The full list can be found `here
<https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connection-prereqs.html#connection-prereqs-get-info-about-instance>`_
under 'Get the user name for your instance'. 'my-instance-public-dns-name' can
be found by clicking on your 'Instances' tab in your EC2 dashboard.
Full connection details can be found in the `AWS guide
<https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_.

.. _prepare_instance:

Prepare the instance
^^^^^^^^^^^^^^^^^^^^

To prepare the instance you need to download the starting kit, data and all
required packages to run submissions. This can be saved as an AMI (image) and
will then be used to launch instances to run submissions.

Below is a basic guide for creating such an AMI manually, using the Iris
challenge. This guide is for an ubuntu instance. If you have a Linux instance
you will not need to add miniconda to your ``~/.profile`` but you will
need to install git.

  - Install miniconda::

        ~ $ LATEST_MINICONDA="http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        ~ $ wget -q $LATEST_MINICONDA -O ~/miniconda.sh
        ~ $ bash ~/miniconda.sh -b
        ~ $ echo '. ${HOME}/miniconda3/etc/profile.d/conda.sh' >> ~/.profile
        ~ $ echo 'conda activate base' >> ~/.profile
        ~ $ source .profile
        ~ $ conda info
        ~ $ conda update --yes --quiet conda pip

  - Get the starting kit material (e.g. for 'iris')::

        ~ $ RAMPKIT_DIR="$HOME/ramp-kits"
        ~ $ project_name="iris"
        ~ $ kit_url="https://github.com/ramp-kits/$project_name"
        ~ $ kit_dir="$RAMPKIT_DIR/$project_name"
        ~ $ git clone $kit_url $kit_dir

  - Update the base conda environment for the needs of the challenge::

        ~ $ environment="$kit_dir/environment.yml"
        ~ $ conda env update --name base --file $environment
        ~ $ conda list

  - Get the data and copy public data to ``ramp-kits/<project>/data/``.
    Note: depending on how ``prepare_data.py`` structures the data files,
    the last command may differ::

        ~ $ RAMPDATA_DIR="$HOME/ramp-data"
        ~ $ data_url="https://github.com/ramp-data/$project_name"
        ~ $ data_dir="$RAMPDATA_DIR/$project_name"
        ~ $ git clone $data_url $data_dir
        ~ $ cd $data_dir
        ~ $ python prepare_data.py
        ~ $ cp data/public/* $kit_dir/data/

  - Test the kit::

        ~ $ cd $kit_dir
        ~ $ ramp-test

Next, save the instance as an AMI. Starting from the instance tab:
Actions -> Image -> Create image. See `Create an AMI from an Amazon EC2
instance <https://docs.aws.amazon.com/toolkit-for-visual-studio/latest/user-guide/tkv-create-ami-from-instance.html>`_
for more details.

Event configuration
^^^^^^^^^^^^^^^

Create an event config.yml (see :ref:`deploy-ramp-event`) and update the
'worker' section, which should look something like::

      ramp:
          problem_name: iris
          event_name: iris_ramp_aws_test
          event_title: "iris ramp aws test"
          event_is_public: true
      worker:
          worker_type: aws
          access_key_id: <aws_access_key_id for boto3 Session>
          secret_access_key: <aws_secret_access_key for boto3 Session>
          region_name: us-west-2 # oregon
          ami_image_name: <name of the AMI set up for this event>
          ami_user_name: ubuntu
          instance_type: t2.micro
          key_name: <name of your pem file, eg iris_key>
          security_group: launch-wizard-103
          key_path: <path to pem file corresponding to user name, eg my_path/iris_key.pem>
          remote_ramp_kit_folder: /home/ubuntu/ramp-kits/iris
          submissions_dir: /home/ramp/ramp_deployment/events/iris_aws/submissions
          predictions_dir: /home/ramp/ramp_deployment/events/iris_aws/predictions
          logs_dir: /home/ramp/ramp_deployment/events/iris_aws/logs
          memory_profiling: false

* ``access_key_id`` and ``secret_access_key``: to create a new access key, go
  to your top navigation bar, click on your user name -> My Security
  Credentials. Open the 'Access keys' tab then click on 'Create New Access
  Key'. You will be prompted to download your 'access key ID' and 'secret
  access key' to a ``.csv`` file, which should be saved in a secure location.
  You can also use an existing access key by obtaining the ID and key from
  your saved ``.csv`` file. See `Managing Access Keys
  <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html#Using_CreateAccessKey>`_
  for more details.
* ``region_name``: zone of your instance, which can be found in the EC2
  console, 'Instances' tab on the left, under 'availability zone'.
* ``ami_image_name``: name you gave to the image you prepared (see
  :ref:`prepare_instance`). This can be found in the EC2 console, under
  'Images' -> 'AMI' tab.
* ``ami_user_name``: user name you used to ssh into your instance.
  Commonly 'ec2-user' or 'ubuntu'.
* ``instance_type``: found in the EC2 console, 'Instances' tab, 'Description'
  tab at the bottom, under 'Instance type'.
* ``key_name``: name of the key, eg if your key file is 'iris_key.pem', the key
  name is 'iris_key'
* ``security_group``: in the EC2 console, 'Instances' tab, 'Description' tab
  at the bottom, under 'Security groups'.
* ``key_path``: path to the you private key used to ssh to your instance
  (see :ref:`launch_aws`). Note that you need to copy your key into the RAMP.
  It is best to give the absolute path. Ensure the permissions of this file
  is set to only 'read' by owner (which can be done using ``chmod 400
  key_name.pem`` command).
* ``remote_ramp_kit_folder``: path to the starting kit folder on
  the AWS image you prepared (see :ref:`prepare_instance`).
* ``submissions_dir``: path to the submissions directory on the RAMP server.
* ``predictions_dir``: path to the predictions directoryon the RAMP server.
* ``logs_dir``: path to store the submission logs on the RAMP server.
* ``memory_profiling``: boolean, whether or not to profile memory used by each
  submission. You need to install `memory profiler
  <https://pypi.org/project/memory-profiler/>`_ in your prepared AMI image
  to enable this.

Create your own worker
----------------------

Currently, the choice of workers in RAMP is quite limited. You might want to
create your own worker for your platform (Openstack, Azure, etc.). This section
illustrates how to implement your own worker.

Your new worker should subclass :class:`ramp_engine.base.BaseWorker`. This
class implements some basic functionalities which you should use.

The ``setup`` and ``teardown`` functions
........................................

The ``setup`` function will initialize the worker before it can be used, e.g.
activate the right conda environment. Your worker should implement this class
and call ``super`` at the end of it. Indeed, the base class is in charge of
updating the status of the worker and logging some information. Thus, your
function should look like::

      def setup(self):
            # implement some initialization for instance launch an
            # Openstack instance
            assert True
            # call the base class to update the status and log
            super().setup()

Similarly, you might need to make some operation to release the worker. Then,
the function ``teardown`` is in charge of this. It should be called similarly
to the ``setup`` function::

      def teardown(self):
            # clean some jobs done by the worker
            # ...
            # call the base class to update the status and log
            super().teardown()

The ``launch_submission`` and ``collect_results`` functions
...........................................................

The actual job of the worker should be implemented in ``launch_submission`` in
charge of running a submission and ``collect_results`` in charge of collecting
and paste them in the location indicated by the dispatcher. As for the other
previous functions, you should call ``super`` at the end of the processing to
update the status of the worker::

      def launch_submission(self):
            # launch ramp test --submission <sub> --save-output on the Openstack
            # instance
            ...
            # call the base class to update the status and log
            super().launch_submission()

Once a submission is trained, the ``ramp test`` command line would store the
results and you should upload those in the directory indicated by the
dispatcher::

      def collect_results(self):
            # the base class will be in charge of checking that the state of
            # the worker is fine
            super().collect_results()
            # write the prediction and logs at the location indicated by the
            # dispatcher (given by the config file)
            log_output = stdout + b'\n\n' + stderr
            log_dir = os.path.join(self.config['logs_dir'],
                                   self.submission)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(os.path.join(log_dir, 'log'), 'wb+') as f:
                f.write(log_output)
            pred_dir = os.path.join(self.config['predictions_dir'],
                                    self.submission)
            output_training_dir = os.path.join(
                self.config['submissions_dir'], self.submission,
                'training_output')
            if os.path.exists(pred_dir):
                shutil.rmtree(pred_dir)
            shutil.copytree(output_training_dir, pred_dir)
            self.status = 'collected'
            logger.info(repr(self))
            return (self._proc.returncode, error_msg)


The ``_is_submission_finished`` function
........................................

You need to implement the ``_is_submission_finished`` function to indicate the
worker that he can call the ``collect_results`` function. This function will
be dependent of the platform that you are using. For instance, for a conda
worker, it would look like::

      def _is_submission_finished(self):
        """Status of the submission.

        The submission was launched in a subprocess. Calling ``poll()`` will
        indicate the status of this subprocess.
        """
        return False if self._proc.poll() is None else True
