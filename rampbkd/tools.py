from __future__ import print_function, absolute_import

import datetime

from .model import Event, Team, EventTeam, Submission


def set_state(event_name, team_name, submission_name, state):
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam.query.filter_by(event=event, team=team).one()
    submission = Submission.query.filter_by(
        name=submission_name, event_team=event_team).one()
    submission.set_state(state)


def get_new_submissions(session, event_name=None, team_name=None):
    if event_name is None:
        new_submissions = Submission.query.filter_by(
            state='new').filter(Submission.is_not_sandbox).order_by(
            Submission.submission_timestamp).all()
    # a fast fix: prefixing event name with 'not_' will exclude the event
    elif event_name[:4] == 'not_':
        event_name = event_name[4:]
        new_submissions = session.query(
            Submission, Event, EventTeam).filter(
            Event.name != event_name).filter(
            Event.id == EventTeam.event_id).filter(
            EventTeam.id == Submission.event_team_id).filter(
            Submission.state == 'new').filter(
            Submission.is_not_sandbox).order_by(
            Submission.submission_timestamp).all()
        if new_submissions:
            new_submissions = list(zip(*new_submissions)[0])
        else:
            new_submissions = []
    else:
        new_submissions = session.query(
            Submission, Event, EventTeam).filter(
            Event.name == event_name).filter(
            Event.id == EventTeam.event_id).filter(
            EventTeam.id == Submission.event_team_id).filter(
            Submission.state == 'new').filter(
            Submission.is_not_sandbox).order_by(
            Submission.submission_timestamp).all()
        if new_submissions:
            new_submissions = list(zip(*new_submissions)[0])
        else:
            new_submissions = []
    # Give ten seconds to upload submission files. Can be eliminated
    # once submission files go into database.
    new_submissions = [
        s for s in new_submissions
        if datetime.datetime.utcnow() - s.submission_timestamp >
        datetime.timedelta(0, 10)]

    if len(new_submissions) == 0:
        return None
    else:
        return new_submissions
