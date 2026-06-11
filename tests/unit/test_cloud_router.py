from fusion_mlx.router.cloud_router import CloudRouter


class TestCloudRouterCircuitBreaker:
     def test_circuit_opens_after_threshold(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=1000)
        assert cr.is_circuit_open() is False

        for _ in range(5):
            cr.report_local_failure()
        assert cr.is_circuit_open() is True

     def test_circuit_closes_on_success(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=1000)
        for _ in range(5):
            cr.report_local_failure()
        assert cr.is_circuit_open() is True
        cr.report_local_success()
        assert cr.is_circuit_open() is False

     def test_should_route_when_circuit_open(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=10000)
        for _ in range(5):
            cr.report_local_failure()
        assert cr.should_route_to_cloud(1) is True

     def test_should_route_by_threshold(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=1000)
        assert cr.should_route_to_cloud(500) is False
        assert cr.should_route_to_cloud(2000) is True

     def test_failure_count_below_threshold(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=1000)
        for _ in range(4):
            cr.report_local_failure()
        assert cr.is_circuit_open() is False

     def test_half_open_after_timeout(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=100000)
        cr._half_open_timeout = 0.1   # 100ms for fast test
        for _ in range(5):
            cr.report_local_failure()
        assert cr.is_circuit_open() is True
        import time
        time.sleep(0.15)
           # After timeout, should_route_to_cloud resets circuit for a probe
        result = cr.should_route_to_cloud(1)
        assert result is False   # half-open: allow probe
        assert cr.is_circuit_open() is False

     def test_half_open_probe_success_closes(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=100000)
        cr._half_open_timeout = 0.05
        for _ in range(5):
            cr.report_local_failure()
        import time
        time.sleep(0.1)
        cr.should_route_to_cloud(1)   # triggers half-open
        cr.report_local_success()
        assert cr.is_circuit_open() is False

     def test_half_open_probe_failure_reopens(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=100000)
        cr._half_open_timeout = 0.05
        cr._circuit_failure_threshold = 1   # one failure to reopen
        for _ in range(5):
            cr.report_local_failure()
        import time
        time.sleep(0.1)
        cr.should_route_to_cloud(1)   # triggers half-open
        cr.report_local_failure()   # probe failed
        assert cr.is_circuit_open() is True

     def test_circuit_open_at_is_recorded(self):
        cr = CloudRouter(cloud_model="gpt-4", threshold=1000)
        assert cr._circuit_open_at is None
        for _ in range(5):
            cr.report_local_failure()
        assert cr._circuit_open_at is not None
        assert isinstance(cr._circuit_open_at, float)
