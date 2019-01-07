import flask_login

from flask import Blueprint
from flask import current_app
from flask import redirect
from flask import session
from flask import url_for

from rampdb.tools.user import add_user_interaction

from frontend import db

mod = Blueprint('auth', __name__)


@mod.route("/login", methods=['GET', 'POST'])
def login():
    add_user_interaction(db.session, interaction='landing')

    # If there is already a user logged in, don't let another log in
    print(current_app.__dict__)
    if flask_login.current_user.is_authenticated:
        session['logged_in'] = True
        return redirect(url_for('problems'))

#     form = LoginForm()
#     if form.validate_on_submit():
#         try:
#             user = User.query.filter_by(name=form.user_name.data).one()
#         except NoResultFound:
#             flash(u'{} does not exist.'.format(form.user_name.data))
#             return redirect(url_for('login'))
#         if not check_password(
#                 form.password.data, user.hashed_password):
#             flash('Wrong password')
#             return redirect(url_for('login'))
#         fl.login_user(user, remember=True)  # , remember=form.remember_me.data)
#         session['logged_in'] = True
#         user.is_authenticated = True
#         db.session.commit()
#         logger.info(u'{} is logged in'.format(fl.current_user.name))
#         add_user_interaction(
#             interaction='login', user=fl.current_user)
#         next = request.args.get('next')
#         if next is None:
#             next = url_for('problems')
#         return redirect(next)

#     return render_template(
#         'login.html',
#         form=form,
#     )