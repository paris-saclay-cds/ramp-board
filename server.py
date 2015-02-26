#!/usr/bin/env python2

if __name__ == "__main__":

    import os
    import databoard.views
    from databoard import app
    from databoard.config_databoard import local_deployment
    
    debug_mode = os.getenv('DEBUGLB', local_deployment)
 
    try: 
        debug_mode = bool(int(debug_mode))
    except ValueError:
        debug_mode = True  # a non empty string means debug
 
    app.run(
        debug=bool(debug_mode), 
        port=os.getenv('SERV_PORT', 8080), 
        host='0.0.0.0')
