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

Running submissions with a worker using Conda environment
---------------------------------------------------------

As previously mentioned, the :class:`ramp_engine.local.CondaEnvWorker` workers
rely on Conda environment to execute the submission.

For instance, if you would like to run the `Iris challenge
<https://github.com/ramp-kits/iris>`_, the starting kit requires numpy, pandas,
and scikit-learn. Thus, the environment used to run the initial submission
should have these libraries installed.

Subsequently, you can create such environment and use it in the event
configuration::

      conda create --name ramp-iris numpy pandas scikit-learn

If you are using a ramp-kit from the `Paris-Saclay CDS
<https://github.com/ramp-kits>`_, each kit will provide either an
``environment.yml`` or ``ami_environment.yml`` file which you can use to create
the environment::

      conda create --name ramp-iris --file environment.yml

Running submissions on Amazon Web Services (AWS)
------------------------------------------------

The AWS worker will train and evaluate the submissions on an AWS instance,
which can be useful if the server itself has not much resources or if you need
specific resources (e.g. GPU's).

You need to create an Amazon Machine Image (AMI) with the starting kit and
data, and the needed packages to run submissions. This AMI will then be used
to launch instances to run submissions.

A very short how-to for creating such an AMI manually:

- launch an Amazon instance
- connect to it
- prepare the instance:

  - Install miniconda::

        LATEST_MINICONDA="http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        wget -q $LATEST_MINICONDA -O ~/miniconda.sh
        bash ~/miniconda.sh -b
        echo '. ${HOME}/miniconda3/etc/profile.d/conda.sh' >> ~/.profile
        echo 'conda activate base' >> ~/.profile
        source .profile
        conda info
        conda update --yes --quiet conda pip

  - Get the starting kit material (example here for Iris)::

        RAMPKIT_DIR="$HOME/ramp-kits"
        project_name="iris"
        kit_url="https://github.com/ramp-kits/$project_name"
        kit_dir="$RAMPKIT_DIR/$project_name"
        git clone $kit_url $kit_dir

  - Update the base conda environment for the needs of the challenge::

        ami_environment="$kit_dir/ami_environment.yml"
        conda env update --name base --file $ami_environment
        conda list

  - Get the data::

        data_dir="$kit_dir/data"
        rm -rf $data_dir && mkdir $data_dir
        git clone https://github.com/ramp-data/iris ramp-data/iris
        cd ramp-data/iris/
        python prepare_data.py
        cd ..

    TODO: figure out this data (in ramp-kits/data or in ramp-data?)

  - Test the kit::

        cd ramp-kits/iris
        ramp-test

- Save the instance as an AMI: from the instance -> Actions -> Image -> Create image

- Create an event config.yml, which should look something like::

      ramp:
          problem_name: iris
          event_name: iris_ramp_aws_test
          event_title: iris_ramp_aws_test
          event_is_public: true
          ...
      worker:
          worker_type: aws
          access_key_id: <aws_access_key_id for boto3 Session>
          secret_access_key: <aws_secret_access_key for boto3 Session>
          region_name: us-west-2 # oregon
          ami_image_name: <name of the AMI set up for this event>
          ami_user_name : ubuntu
          instance_type : t2.micro
          key_name: <user that can ssh connect with the created AWS instance>
          security_group : launch-wizard-103
          key_path: <path to pem file corresponding to user name>
          remote_ramp_kit_folder : /home/ubuntu/ramp-kits/iris
          memory_profiling : false

Create your own worker
----------------------

Currently, the choice of workers in RAMP is quite limited. You might want to
create your own worker for your platform (Openstack, Azure, etc.). This section
illustrates how to implement your own worker.

Your new worker should subclass :class:`ramp_engine.base.BaseWorker`. This
class implement some basic functionalities which you should use.

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
            super().setup(self)

Similarly, you might need to make some operation to release the worker. Then,
the function ``teardown`` is in charge of this. It should be called similarly
to the ``setup`` function::

      def teardown(self):
            # clean some jobs done by the worker
            # ...
            # call the base class to update the status and log
            super().teardown(self)

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
            super().launch_submission(self)

Once a submission is trained, the ``ramp test`` command line would store the
results and you should upload those in the directory indicated by the
dispatcher::

      def collect_results(self):
            # the base class will be in charge of checking that the state of
            # the worker is fine
            super().collect_results(self)
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
