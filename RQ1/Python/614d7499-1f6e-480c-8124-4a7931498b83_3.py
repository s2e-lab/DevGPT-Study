from unittest.mock import MagicMock

def test_query_atoms():
    mock_config = MagicMock()
    mock_config.access_key = "test_key"
    mock_config.endpoint_suffix = "test_suffix"
    mock_config.account_name = "test_name"
    mock_config.connection_string = "test_connection_string"

    query_instance = SampleTablesQuery(mock_config)
    # your test code here
