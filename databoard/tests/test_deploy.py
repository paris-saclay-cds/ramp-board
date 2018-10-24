from databoard.deploy import deploy
import databoard.db_tools as db_tools


def test_deploy():
    deploy()


def test_add_users():
    db_tools.create_user(
        name='kegl', password='pwd',
        lastname='Kegl', firstname='Balazs',
        email='balazs.kegl@gmail.com', access_level='admin')
    db_tools.create_user(
        name='agramfort', password='pwd',
        lastname='Gramfort', firstname='Alexandre',
        email='alexandre.gramfort@gmail.com', access_level='admin')
    db_tools.create_user(
        name='akazakci', password='pwd',
        lastname='Akin', firstname='Kazakci',
        email='osmanakin@gmail.com', access_level='admin')
    db_tools.create_user(
        name='mcherti', password='pwd', lastname='Cherti',
        firstname='Mehdi', email='mehdicherti@gmail.com',
        access_level='admin')
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
