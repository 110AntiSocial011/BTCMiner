import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.metrics import MetricsService
from prometheus_client import Counter

@pytest.fixture(autouse=True)
def mock_prometheus_server(mocker):
    mocker.patch('prometheus_client.start_http_server')

def test_metrics_initialization():
    metrics = MetricsService(8000)
    assert isinstance(metrics.addresses_checked, Counter)
    assert isinstance(metrics.balance_hits, Counter)
    assert isinstance(metrics.api_errors, Counter)
