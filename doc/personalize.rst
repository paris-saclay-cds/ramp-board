##############################
Personalize your RAMP instance
##############################

It is possible to personalize your RAMP front page.

Front page
----------
You might be interested in displaying some images (logos) in the ``Powered by``
section in the very bottom of your RAMP webpage.

To do so, add all the images you wish to have displayed to the following
directory::

    ~ $ ramp-board/ramp-frontend/ramp_frontend/static/img/powered_by

Your images must have one of the following extensions:

* .png
* .jpg (.jpeg)
* .gif
* .svg

When you reload your RAMP page the new ``Powered by`` section should appear.


Privacy Policy page
-------------------

You can add an optional Privacy Policy page, setting the following in the main
`config.yaml`,

.. code::

    flask:
        ...
        privacy_policy_page: "<path.html>"

Where the ``privacy_policy_page`` can be either a path to an HTML file, or
directly the HTML contents of that page.

This will enable the ``/privacy_policy`` endpoint, and will add it to the footer
menu.


Sign up and login pages
-----------------------

You can add personalized messages to the Sign Up and Login pages, as follows,

.. code::

    flask:
        ...
        login_instructions: "instructions A"
        sign_up_instructions: "instructions A"
        sign_up_ask_social_media: True    # ask for social media acounts (optional)

where ``login_instructions`` and ``sign_up_instructions`` can be either a path to an HTML
file, or directly the HTML contents.

By including HTML code with JavaScript, these field can also be used to customize the
Sign Up and Login forms.
