from prometheus_client import Counter, Gauge, Histogram, start_http_server

class MetricsService:
    def __init__(self, port: int):
        # Make sure metric names match exactly
        self.addresses_checked = Counter('addresses_checked_total', 'Total addresses checked')
        self.balance_hits = Counter('balance_hits_total', 'Total addresses with balance')
        self.api_errors = Counter('api_errors_total', 'Total API errors')
        self.active_threads = Gauge('active_threads', 'Number of active checking threads')
        self.balance_check_duration = Histogram('balance_check_duration_seconds', 'Time spent checking balance')
        self.memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
        self.api_requests = Counter('api_requests_total', 'Total API requests')
        self.api_latency = Histogram('api_latency_seconds', 'API request latency')
        self.valid_addresses = Counter('valid_addresses_total', 'Total valid addresses')
        self.invalid_addresses = Counter('invalid_addresses_total', 'Total invalid addresses')
        try:
            start_http_server(port)
        except Exception as e:
            print(f"Warning: Could not start metrics server on port {port}: {e}")
