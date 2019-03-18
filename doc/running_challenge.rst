###############################
Useful tips when running a RAMP
###############################

When running a RAMP challenge, you might need one of the following tips.

How to restart a failed submission manually
-------------------------------------------

If for some reason, one of the submission failed and you would like to
re-evaluate this submission, you should change the state of this submission.
You can use the following command to change the status of a submission::

    ramp database set-submission-state --submission-id <id> --state new

Since the submission was set to ``new``, the RAMP dispatcher will automatically
pick up this submission to train it again.
