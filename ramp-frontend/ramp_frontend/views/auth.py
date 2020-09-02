"""Blueprint for all authentication functions for the RAMP frontend."""
import logging

import flask_login

from flask import abort
from flask import Blueprint
from flask import current_app as app
from flask import flash
from flask import redirect
from flask import request
from flask import render_template
from flask import session
from flask import url_for

from itsdangerous import URLSafeTimedSerializer

from ramp_database.utils import check_password
from ramp_database.utils import hash_password

from ramp_database.tools.user import add_user
from ramp_database.tools.user import add_user_interaction
from ramp_database.tools.user import get_user_by_name_or_email
from ramp_database.tools.user import set_user_by_instance

from ramp_database.model import User

from ramp_database.exceptions import NameClashError

from ramp_frontend import db
from ramp_frontend import login_manager

from ..forms import EmailForm
from ..forms import LoginForm
from ..forms import PasswordForm
from ..forms import UserCreateProfileForm
from ..forms import UserUpdateProfileForm

from ..utils import body_formatter_user
from ..utils import send_mail

logger = logging.getLogger('RAMP-FRONTEND')
mod = Blueprint('auth', __name__)
ts = URLSafeTimedSerializer(app.config["SECRET_KEY"])


@login_manager.user_loader
def load_user(id):
    """Load a user in the login manager.

    This function is used by Flask-Login to manage the current-user connection.

    Parameters
    ----------
    id : int
        The user ID.
    """
    return User.query.get(id)


@mod.route("/login", methods=['GET', 'POST'])
def login():
    """Login request."""
    if app.config['TRACK_USER_INTERACTION']:
        add_user_interaction(db.session, interaction='landing')

    if flask_login.current_user.is_authenticated:
        logger.info('User already logged-in')
        session['logged_in'] = True
        return redirect(url_for('ramp.problems'))

    form = LoginForm()
    if form.validate_on_submit():
        user = get_user_by_name_or_email(db.session,
                                         name=form.user_name.data)
        if user is None:
            msg = 'User "{}" does not exist'.format(form.user_name.data)
            flash(msg)
            logger.info(msg)
            return redirect(url_for('auth.login'))
        if not check_password(form.password.data,
                              user.hashed_password):
            msg = 'Wrong password'
            flash(msg)
            logger.info(msg)
            return redirect(url_for('auth.login'))
        flask_login.login_user(user, remember=True)
        session['logged_in'] = True
        user.is_authenticated = True
        db.session.commit()
        logger.info('User "{}" is logged in'
                    .format(flask_login.current_user.name))
        if app.config['TRACK_USER_INTERACTION']:
            add_user_interaction(
                db.session, interaction='login', user=flask_login.current_user
            )
        next_ = request.args.get('next')
        if next_ is None:
            next_ = url_for('ramp.problems')
        return redirect(next_)

    return render_template('login.html', form=form)


@mod.route("/logout")
@flask_login.login_required
def logout():
    """Logout request."""
    user = flask_login.current_user
    if app.config['TRACK_USER_INTERACTION']:
        add_user_interaction(db.session, interaction='logout', user=user)
    session['logged_in'] = False
    user.is_authenticated = False
    db.session.commit()
    logger.info('{} is logged out'.format(user))
    flask_login.logout_user()

    return redirect(url_for('auth.login'))


@mod.route("/sign_up", methods=['GET', 'POST'])
def sign_up():
    """Sign-up request."""
    if flask_login.current_user.is_authenticated:
        session['logged_in'] = True
        return redirect(url_for('ramp.problems'))

    form = UserCreateProfileForm()
    if form.validate_on_submit():
        try:
            user = add_user(
                session=db.session,
                name=form.user_name.data,
                password=form.password.data,
                lastname=form.lastname.data,
                firstname=form.firstname.data,
                email=form.email.data,
                linkedin_url=form.linkedin_url.data,
                twitter_url=form.twitter_url.data,
                facebook_url=form.facebook_url.data,
                google_url=form.google_url.data,
                github_url=form.github_url.data,
                website_url=form.website_url.data,
                bio=form.bio.data,
                is_want_news=form.is_want_news.data,
                access_level='not_confirmed'
            )
        except NameClashError as e:
            flash(str(e))
            logger.info(str(e))
            return redirect(url_for('auth.sign_up'))
        # send an email to the participant such that he can confirm his email
        token = ts.dumps(user.email)
        recover_url = url_for(
            'auth.user_confirm_email', token=token, _external=True
        )
        subject = "Confirm your email for signing-up to RAMP"
        body = ('Hi {}, \n\n Click on the following link to confirm your email'
                ' address and finalize your sign-up to RAMP.\n\n Note that '
                'your account still needs to be approved by a RAMP '
                'administrator.\n\n'
                .format(user.firstname))
        body += recover_url
        body += '\n\nSee you on the RAMP website!'
        send_mail(user.email, subject, body)
        logger.info(
            '{} has signed-up to RAMP'.format(user.name)
        )
        flash(
            "We sent a confirmation email. Go read your email and click on "
            "the confirmation link"
        )
        return redirect(url_for('auth.login'))
    return render_template('sign_up.html', form=form)


@mod.route("/update_profile", methods=['GET', 'POST'])
@flask_login.login_required
def update_profile():
    """User profile update."""
    form = UserUpdateProfileForm()
    form.user_name.data = flask_login.current_user.name
    if form.validate_on_submit():
        set_user_by_instance(
            db.session,
            user=flask_login.current_user,
            lastname=form.lastname.data,
            firstname=form.firstname.data,
            email=form.email.data,
            linkedin_url=form.linkedin_url.data,
            twitter_url=form.twitter_url.data,
            facebook_url=form.facebook_url.data,
            google_url=form.google_url.data,
            github_url=form.github_url.data,
            website_url=form.website_url.data,
            is_want_news=form.is_want_news.data
        )
        # send_register_request_mail(user)
        return redirect(url_for('ramp.problems'))
    form.lastname.data = flask_login.current_user.lastname
    form.firstname.data = flask_login.current_user.firstname
    form.email.data = flask_login.current_user.email
    form.linkedin_url.data = flask_login.current_user.linkedin_url
    form.twitter_url.data = flask_login.current_user.twitter_url
    form.facebook_url.data = flask_login.current_user.facebook_url
    form.google_url.data = flask_login.current_user.google_url
    form.github_url.data = flask_login.current_user.github_url
    form.website_url.data = flask_login.current_user.website_url
    form.bio.data = flask_login.current_user.bio
    form.is_want_news.data = flask_login.current_user.is_want_news
    return render_template('update_profile.html', form=form)


@mod.route('/reset_password', methods=["GET", "POST"])
def reset_password():
    """Reset password of a RAMP user."""
    form = EmailForm()
    error = ''
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).one_or_none()
        if user and user.access_level != 'asked':
            token = ts.dumps(user.email)
            recover_url = url_for(
                'auth.reset_with_token', token=token, _external=True
            )

            subject = "Password reset requested - RAMP website"
            body = ('Hi {}, \n\nclick on the link to reset your password:\n'
                    .format(user.firstname))
            body += recover_url
            body += '\n\nSee you on the RAMP website!'
            send_mail(user.email, subject, body)
            logger.info(
                'Password reset requested for user {}'.format(user.name)
            )
            logger.info(recover_url)
            flash('An email to reset your password has been sent')
            return redirect(url_for('auth.login'))
        elif user is None:
            error = ('The email address is not linked to any user. You can '
                     'sign-up instead.')
        else:
            error = ('Your account has not been yet approved. You cannot '
                     'change the password already.')
    return render_template('reset_password.html', form=form, error=error)


@mod.route('/reset/<token>', methods=["GET", "POST"])
def reset_with_token(token):
    """Reset password by passing a token (email).

    Parameters
    ----------
    token : str
        The token associated with an email address.
    """
    try:
        email = ts.loads(token, max_age=86400)
    except Exception as e:
        logger.error(str(e))
        abort(404)

    form = PasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=email).one_or_none()
        if user is None:
            logger.error('The error was deleted before resetting his/her '
                         'password')
            abort(404)
        (User.query.filter_by(email=email)
                   .update({
                       "hashed_password":
                       hash_password(form.password.data).decode()}))
        db.session.commit()
        return redirect(url_for('auth.login'))

    return render_template('reset_with_token.html', form=form, token=token)


@mod.route('/confirm_email/<token>', methods=["GET", "POST"])
def user_confirm_email(token):
    """Confirm a user account using his email address and a token to approve.

    Parameters
    ----------
    token : str
        The token associated with an email address.
    """
    try:
        email = ts.loads(token, max_age=86400)
    except Exception as e:
        logger.error(str(e))
        abort(404)

    user = User.query.filter_by(email=email).one_or_none()
    if user is None:
        flash(
            'You did not sign-up yet to RAMP. Please sign-up first.',
            category='error'
        )
        return redirect(url_for('auth.sign_up'))
    elif user.access_level in ('user', 'admin'):
        flash(
            "Your account is already approved. You don't need to confirm your "
            "email address", category='error'
        )
        return redirect(url_for('auth.login'))
    elif user.access_level == 'asked':
        flash(
            "Your email address already has been confirmed. You need to wait "
            "for an approval from a RAMP administrator", category='error'
        )
        return redirect(url_for('general.index'))
    User.query.filter_by(email=email).update({'access_level': 'asked'})
    db.session.commit()
    admin_users = User.query.filter_by(access_level='admin')
    for admin in admin_users:
        subject = 'Approve registration of {}'.format(
            user.name
        )
        body = body_formatter_user(user)
        url_approve = ('http://{}/sign_up/{}'
                       .format(app.config['DOMAIN_NAME'], user.name))
        body += 'Click on the link to approve the registration '
        body += 'of this user: {}'.format(url_approve)
        send_mail(admin.email, subject, body)
    flash(
        "An email has been sent to the RAMP administrator(s) who will "
        "approve your account"
    )
    return redirect(url_for('auth.login'))
