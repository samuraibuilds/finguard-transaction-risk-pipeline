from fastapi.testclient import TestClient
from finguard.api.app import app
def test_health_reports_service_state():
    with TestClient(app) as c:
        r=c.get('/health')
        assert r.status_code==200
        assert r.json()['status'] in ('ok','degraded')
        assert 'model_loaded' in r.json()
