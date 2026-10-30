import pytest
from finguard.common import config
def test_config_loads(): assert config()["project"]["batch_id"]=="demo-batch-001"
def test_missing_config_fails():
    with pytest.raises(FileNotFoundError): config("missing.yaml")
