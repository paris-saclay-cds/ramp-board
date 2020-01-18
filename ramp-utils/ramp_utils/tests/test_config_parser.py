import tempfile
import yaml
import pytest

from ramp_utils import read_config


@pytest.fixture
def simple_config(database_connection):
    data = {'sqlalchemy': {'username': 'mrramp', 'password': 'mrramp'},
            'ramp': {'event_name': 'iris_test'}}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml') as config_file:
        yaml.dump(data, config_file, default_flow_style=False)
        yield config_file.name


@pytest.mark.parametrize(
    "filter_section, expected_config",
    [(None, {'sqlalchemy': {'username': 'mrramp', 'password': 'mrramp'},
             'ramp': {'event_name': 'iris_test'}}),
     (['ramp'], {'ramp': {'event_name': 'iris_test'}}),
     ('ramp', {'event_name': 'iris_test'})]
)
def test_read_config_filtering(simple_config, filter_section, expected_config):
    config = read_config(simple_config, filter_section=filter_section,
                         check_requirements=False)
    assert config == expected_config


@pytest.mark.parametrize(
    "filter_section, check_requirements, err_msg",
    [('unknown', False, 'The section "unknown" is not in'),
     (None, True, 'The section "sqlalchemy" in the')]
)
def test_read_config_error(simple_config, filter_section, check_requirements,
                           err_msg):
    with pytest.raises(ValueError, match=err_msg):
        read_config(simple_config, filter_section=filter_section,
                    check_requirements=check_requirements)
