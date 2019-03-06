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


Running a submissions on Amazon Web Services (AWS)
--------------------------------------------------

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
          event: iris
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
          local_predictions_folder : <...>
          local_log_folder : <...>
          memory_profiling : false
