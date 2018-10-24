from databoard.deploy import deploy
import databoard.db_tools as db_tools


def test_deploy():
    deploy()


def test_add_users():
    db_tools.create_user(
        name='test_user', password='test',
        lastname='Test', firstname='User',
        email='test.user@gmail.com', access_level='user')
    db_tools.create_user(
        name='test_iris_admin', password='test',
        lastname='Admin', firstname='Iris',
        email='iris.admin@gmail.com', access_level='user')


def test_setup_workflows():
    db_tools.setup_workflows()


def _add_problem_and_event(problem_name, test_user_name, with_download='True'):
    db_tools.add_problem(
        problem_name, with_download=with_download, force='True')
    event_name = '{}'.format(problem_name)
    event_title = 'test event'
    db_tools.add_event(
        problem_name, event_name, event_title, is_public='True', force='True')
    db_tools.sign_up_team(event_name, test_user_name)
    db_tools.submit_starting_kit(event_name, test_user_name)
    submissions = db_tools.get_submissions(event_name, test_user_name)
    db_tools.train_test_submissions(
        submissions, force_retrain_test=True, is_parallelize=False)
    db_tools.compute_contributivity(event_name)
    db_tools.update_leaderboards(event_name)
    db_tools.update_user_leaderboards(event_name, test_user_name)
    db_tools.compute_contributivity(event_name)


def test_add_problem_and_event():
    _add_problem_and_event('iris', 'test_user')
    _add_problem_and_event('boston_housing', 'test_user')


def test_add_keywords():
    import databoard.db_tools as db_tools
    db_tools.add_keyword('botany', 'data_domain', 'scientific data', 'Botany.')
    db_tools.add_keyword(
        'real estate', 'data_domain', 'industrial data', 'Real estate.')
    db_tools.add_keyword(
        'regression', 'data_science_theme', None, 'Regression.')
    db_tools.add_keyword(
        'classification', 'data_science_theme', None, 'Classification.')
    db_tools.add_problem_keyword('iris', 'classification')
    db_tools.add_problem_keyword('iris', 'botany')
    db_tools.add_problem_keyword('boston_housing', 'regression')
    db_tools.add_problem_keyword('boston_housing', 'real estate')
