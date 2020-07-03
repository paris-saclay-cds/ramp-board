from collections import deque
import logging
import os
import signal
import subprocess
import time

from ramp_database.model import Event
from ramp_database.utils import session_scope

from ramp_utils import read_config

logger = logging.getLogger("RAMP-DAEMON")


class Daemon:
    """RAMP daemon starting dispatchers for open challenges.

    Parameters
    ----------
    config : str
        Path to the configuration YAML file containing the information about
        the database.
    events_dir : str
        The path in which all events configuration files will be located. We
        expect a pattern as `event_dir/<a ramp event>/config.yml`. The config
        file will be used to start the daemon.
    """

    def __init__(self, config, events_dir):
        self.config = config
        self._database_config = read_config(
            config, filter_section="sqlalchemy"
        )
        self.events_dir = os.path.abspath(events_dir)
        if not os.path.isdir(self.events_dir):
            raise ValueError(
                "The path {} is not existing.".format(events_dir)
            )
        self._proc = deque()
        signal.signal(signal.SIGINT, self.kill_dispatcher)
        signal.signal(signal.SIGTERM, self.kill_dispatcher)
        self._poison_pill = False

    def launch_dispatchers(self, session):
        events = [e for e in session.query(Event).all() if e.is_open]
        for e in events:
            event_config = os.path.join(
                self.events_dir, e.name, "config.yml"
            )
            cmd_dispatcher = [
                "ramp-launch", "dispatcher",
                "--config", self.config,
                "--event-config", event_config,
                "--verbose"
            ]
            proc = subprocess.Popen(
                cmd_dispatcher,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._proc.append((e.name, proc))
            logger.info(
                "Launch dispatcher for the event {}".format(e.name)
            )

    def kill_dispatcher(self, signum, frame):
        while len(self._proc) != 0:
            event, proc = self._proc.pop()
            proc.kill()
            logger.info(
                "Kill dispatcher for the event {}".format(event)
            )
        self._poison_pill = True

    def launch(self):
        """Start the daemon.

        The daemon will be killed using a keyboard interuption.
        """
        with session_scope(self._database_config) as session:
            self.launch_dispatchers(session)
            while not self._poison_pill:
                time.sleep(5)
