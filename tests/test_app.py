"""Smoke tests for the Flask application and saved local artifacts."""


def test_health_endpoint_reports_loaded_artifacts():
    from app.run import app

    response = app.test_client().get('/health')
    payload = response.get_json()

    assert response.status_code == 200
    assert payload['status'] == 'ok'
    assert payload['rows'] > 20_000
    assert payload['trained_categories'] == 35


def test_empty_query_is_rejected():
    from app.run import app

    response = app.test_client().get('/go?query=')

    assert response.status_code == 400
    assert b'Enter a disaster-response message' in response.data


def test_prediction_route_returns_classification_page():
    from app.run import app

    response = app.test_client().get('/go?query=We+need+clean+water+and+food')

    assert response.status_code == 200
    assert b'Classification Results' in response.data
    assert b'Water' in response.data
