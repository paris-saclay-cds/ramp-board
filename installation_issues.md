
# "ImportError: cannot import name _remove_dead_weakref" in the logs after starting apache2 server

This happens if there is a mismatch between python version and the python version for which apache2 wsgi
module has been compiled with, so it could happen if python is updated for instance. 
Make sure the versions match. To know python version for which
wsgi was compiled with, search in the log: 

    cat /var/log/apache2/error.log |grep wsgi

you will find:

"[Fri Apr 20 06:25:03.688214 2018] [wsgi:warn] [pid 4083:tid 139699071551360] mod_wsgi: Compiled for Python/2.7.11."

One solution would be to downgrade python version to the one which corresponds to wsgi, in this case:

    conda install python==2.7.11
