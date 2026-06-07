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
