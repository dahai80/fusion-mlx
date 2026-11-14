import json

import pytest
from fastapi import HTTPException

from fusion_mlx.admin.fine_tune_route import (
    _ring_hostfile,
    _validate_distributed_body,
    _write_hostfile_temp,
)
from fusion_mlx.training.service import (
    FineTuneJob,
    restore_dist_env,
    stage_dist_env,
)


class TestRingHostfile:
    def test_single_node_is_empty(self):
        assert _ring_hostfile([{"host": "10.0.0.1"}], 45100) == ""

    def test_two_ranks_list_of_lists(self):
        nodes = [{"host": "10.0.0.1"}, {"host": "10.0.0.2"}]
        hf = _ring_hostfile(nodes, 45100)
        parsed = json.loads(hf)
        assert parsed == [["10.0.0.1:45100"], ["10.0.0.2:45101"]]

    def test_ports_increment_per_rank(self):
        nodes = [{"host": "1.2.3.4"}] * 3
        parsed = json.loads(_ring_hostfile(nodes, 46000))
        assert [p[0] for p in parsed] == [
            "1.2.3.4:46000",
            "1.2.3.4:46001",
            "1.2.3.4:46002",
        ]


class TestHostfileTemp:
    def test_roundtrip(self, tmp_path):
        path = _write_hostfile_temp('[["1.1.1.1:45000"]]', "j1")
        try:
            with open(path) as f:
                assert json.load(f) == [["1.1.1.1:45000"]]
        finally:
            import os

            os.unlink(path)


class TestValidateDistributedBody:
    def _body(self, **over):
        body = {
            "model_id": "org/m",
            "nodes": [
                {"host": "127.0.0.1", "port": 11434, "dataset": "/d/a.jsonl"},
                {"host": "127.0.0.2", "port": 11434, "dataset": "/d/b.jsonl"},
            ],
            "config": {"iters": 2},
        }
        body.update(over)
        return body

    def test_valid_body(self):
        model_id, config, nodes, port = _validate_distributed_body(self._body())
        assert model_id == "org/m"
        assert len(nodes) == 2
        assert config.iters == 2
        assert port == 45100

    def test_missing_model_id(self):
        with pytest.raises(HTTPException) as e:
            _validate_distributed_body(self._body(model_id=""))
        assert e.value.status_code == 400

    def test_missing_nodes(self):
        with pytest.raises(HTTPException) as e:
            _validate_distributed_body(self._body(nodes=[]))
        assert "nodes" in e.value.detail

    def test_node_without_dataset(self):
        bad = [{"host": "h", "port": 1}]
        with pytest.raises(HTTPException) as e:
            _validate_distributed_body(self._body(nodes=bad))
        assert "dataset" in e.value.detail

    def test_bad_config_rejected(self):
        with pytest.raises(HTTPException) as e:
            _validate_distributed_body(self._body(config={"fine_tune_type": "bogus"}))
        assert e.value.status_code == 400


class TestDistEnvStaging:
    def test_stage_and_restore_roundtrip(self, monkeypatch):
        monkeypatch.setenv("MLX_RANK", "7")
        saved = stage_dist_env({"rank": 0, "world_size": 2, "hostfile": "/tmp/hf.json"})
        try:
            import os

            assert os.environ["MLX_RANK"] == "0"
            assert os.environ["MLX_WORLD_SIZE"] == "2"
            assert os.environ["MLX_HOSTFILE"] == "/tmp/hf.json"
        finally:
            restore_dist_env(saved)
        import os

        assert os.environ["MLX_RANK"] == "7"
        assert "MLX_WORLD_SIZE" not in os.environ
        assert "MLX_HOSTFILE" not in os.environ

    def test_stage_none_is_noop(self):
        assert stage_dist_env(None) == {}

    def test_stage_without_hostfile_sets_rank_only(self, monkeypatch):
        monkeypatch.delenv("MLX_HOSTFILE", raising=False)
        saved = stage_dist_env({"rank": 1, "world_size": 3})
        try:
            import os

            assert os.environ["MLX_RANK"] == "1"
            assert "MLX_HOSTFILE" not in os.environ
        finally:
            restore_dist_env(saved)


class TestJobDistributedField:
    def test_default_none_and_dict_roundtrip(self):
        job = FineTuneJob(model_id="m", dataset="d")
        assert job.distributed is None
        d = job.to_dict()
        assert d["distributed"] is None

        job2 = FineTuneJob(
            model_id="m",
            dataset="d",
            distributed={"role": "worker", "rank": 1, "world_size": 2},
        )
        assert job2.to_dict()["distributed"]["role"] == "worker"

    def test_create_job_passes_distributed_and_job_id(self):
        from fusion_mlx.training.service import FineTuneService

        svc = FineTuneService.__new__(FineTuneService)
        svc._jobs = {}
        svc._queue = []
        svc._current_job_id = None
        svc._persist_jobs = lambda: None
        job = svc.create_job(
            model_id="org/m",
            dataset="/d/a",
            job_id="shared123",
            distributed={"role": "worker", "rank": 1, "world_size": 2},
        )
        assert job.job_id == "shared123"
        assert job.distributed["rank"] == 1


class TestWorkerEndpoint:
    def test_worker_endpoint_creates_shared_job(self):
        import asyncio

        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from fusion_mlx.training.service import FineTuneService

        app = FastAPI()
        from fusion_mlx.admin import fine_tune_route as ftr

        svc = FineTuneService.__new__(FineTuneService)
        svc._jobs = {}
        svc._queue = []
        svc._current_job_id = None
        svc._persist_jobs = lambda: None
        svc.start_processing = lambda: None
        ftr._fine_tune_service = svc
        app.include_router(ftr._router, prefix="/admin")
        from fusion_mlx.admin.auth import require_admin

        app.dependency_overrides[require_admin] = lambda: True

        async def run():
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://t"
            ) as client:
                resp = await client.post(
                    "/admin/api/fine-tune/distributed/worker",
                    json={
                        "job_id": "abc123",
                        "model_id": "org/m",
                        "config": {"iters": 1},
                        "dataset": "/d/b.jsonl",
                        "rank": 1,
                        "world_size": 2,
                        "hostfile_json": '[["127.0.0.1:45100"],["127.0.0.1:45101"]]',
                        "adapter_name": "lora-x",
                    },
                )
                assert resp.status_code == 200, resp.text
                data = resp.json()
                assert data["job_id"] == "abc123"
                assert data["distributed"]["role"] == "worker"
                assert data["distributed"]["rank"] == 1
                assert data["distributed"]["hostfile"].endswith(".json")

        asyncio.run(run())

    def test_worker_requires_rank_and_world(self):
        import asyncio

        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from fusion_mlx.training.service import FineTuneService

        app = FastAPI()
        from fusion_mlx.admin import fine_tune_route as ftr

        svc = FineTuneService.__new__(FineTuneService)
        svc._jobs = {}
        svc._queue = []
        svc._current_job_id = None
        svc._persist_jobs = lambda: None
        ftr._fine_tune_service = svc
        app.include_router(ftr._router, prefix="/admin")
        from fusion_mlx.admin.auth import require_admin

        app.dependency_overrides[require_admin] = lambda: True

        async def run():
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://t"
            ) as client:
                resp = await client.post(
                    "/admin/api/fine-tune/distributed/worker",
                    json={"model_id": "m", "dataset": "/d"},
                )
                assert resp.status_code == 400

        asyncio.run(run())
