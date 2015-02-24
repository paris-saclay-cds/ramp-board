#!/usr/bin/env python2

import os

if __name__ == "__main__":

    from databoard import app
    from databoard.config_databoard import (
        serve_port,
        local_deployment,
    )

    debug_mode = os.environ.get('DEBUGLB', local_deployment)
    try: 
        debug_mode = bool(int(debug_mode))
    except ValueError:
        debug_mode = True  # a non empty string means debug
    app.run(debug=bool(debug_mode), port=serve_port, host='0.0.0.0')
