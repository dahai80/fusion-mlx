# SPDX-License-Identifier: Apache-2.0
# Regression tests for the third M-series batch of
# audit/fusion-mlx-codeing-audit-report-1019.md:
#   M22 bounded streaming deque drops silently, M23 O(n^2) aggregate merge.

from __future__ import annotations

import logging

import pytest

from fusion_mlx.output_collector import RequestOutputCollector
from fusion_mlx.request import RequestOutput


@pytest.fixture(autouse=True)
def _reset_totals():
    RequestOutputCollector._dropped_outputs_total = 0
    RequestOutputCollector._dropped_tokens_total = 0
    yield


class TestStreamDequeDropIsVisible:
    # M22: deque(maxlen=...) used to discard the oldest output with no counter
    # and no log — the client saw a stream with missing tokens.

    def _fill(self, monkeypatch, cap=3):
        monkeypatch.setattr(RequestOutputCollector, "_STREAM_DEQUE_MAXLEN", cap)
        return RequestOutputCollector(aggregate=False)

    def test_drop_counted_and_logged(self, monkeypatch, caplog):
        collector = self._fill(monkeypatch)
        with caplog.at_level(logging.ERROR, logger="fusion_mlx.output_collector"):
            for i in range(5):
                collector.put(
                    RequestOutput(request_id="r1", new_token_ids=[i], new_text=str(i))
                )

        # Newest outputs survive, oldest two were dropped.
        assert [o.new_token_ids[0] for o in collector._queue] == [2, 3, 4]
        assert collector._dropped_outputs == 2
        assert collector._dropped_tokens == 2
        assert RequestOutputCollector.dropped_stats() == {
            "dropped_outputs_total": 2,
            "dropped_tokens_total": 2,
        }
        dropped = [r for r in caplog.records if "deque full" in r.message]
        assert dropped, "drop must be logged"
        assert dropped[0].levelno == logging.ERROR

    def test_drop_log_is_throttled(self, monkeypatch, caplog):
        collector = self._fill(monkeypatch, cap=1)
        with caplog.at_level(logging.ERROR, logger="fusion_mlx.output_collector"):
            for i in range(64):
                collector.put(RequestOutput(request_id="r", new_token_ids=[i]))

        # Powers of two only: 1, 2, 4, 8, 16, 32.
        dropped = [r for r in caplog.records if "deque full" in r.message]
        assert len(dropped) == 6
        assert collector._dropped_outputs == 63

    def test_delivered_output_carries_drop_count(self, monkeypatch):
        collector = self._fill(monkeypatch, cap=2)
        for i in range(5):
            collector.put(RequestOutput(request_id="r", new_token_ids=[i]))

        first = collector.get_nowait()
        assert first.dropped_outputs == 3
        assert first.new_token_ids == [3]
        # Later deliveries keep reporting the accumulated loss.
        assert collector.get_nowait().dropped_outputs == 3

    def test_no_drop_below_capacity(self, monkeypatch):
        collector = self._fill(monkeypatch, cap=4)
        for i in range(4):
            collector.put(RequestOutput(request_id="r", new_token_ids=[i]))

        assert collector._dropped_outputs == 0
        assert RequestOutputCollector.dropped_stats()["dropped_outputs_total"] == 0


class TestAggregateMergeIsLinear:
    # M23: put() used to rebuild the merged RequestOutput (token list + text
    # copy) on every step — O(n^2) over a long generation.

    def test_token_buffer_is_extended_in_place(self):
        collector = RequestOutputCollector(aggregate=True)
        collector.put(RequestOutput(request_id="r", new_token_ids=[0], new_text="a"))
        buffer_id = id(collector._acc_token_ids)

        for i in range(1, 500):
            collector.put(
                RequestOutput(request_id="r", new_token_ids=[i], new_text="x")
            )

        assert id(collector._acc_token_ids) == buffer_id
        assert len(collector._acc_token_ids) == 500

    def test_merged_result_carries_everything(self):
        collector = RequestOutputCollector(aggregate=True)
        collector.put(
            RequestOutput(
                request_id="r",
                new_token_ids=[1, 2],
                new_text="Hello",
                prompt_tokens=7,
                generated_at=1.0,
                time_to_first_token=0.5,
                logprobs=[{"token": "Hello", "logprob": -0.1}],
            )
        )
        collector.put(
            RequestOutput(
                request_id="r",
                new_token_ids=[3],
                new_text=" world",
                completion_tokens=3,
                finished=True,
                finish_reason="stop",
                generated_until=2.0,
                logprobs=[{"token": " world", "logprob": -0.2}],
            )
        )

        merged = collector.get_nowait()
        assert merged is not None
        assert merged.new_token_ids == [1, 2, 3]
        assert merged.new_text == "Hello world"
        assert merged.completion_tokens == 3
        assert merged.finished is True
        assert merged.finish_reason == "stop"
        # Sticky fields keep the first observation.
        assert merged.generated_at == 1.0
        assert merged.time_to_first_token == 0.5
        assert merged.generated_until == 2.0
        assert [lp["token"] for lp in merged.logprobs] == ["Hello", " world"]

    def test_second_window_starts_clean(self):
        collector = RequestOutputCollector(aggregate=True)
        collector.put(RequestOutput(request_id="r", new_token_ids=[1], new_text="a"))
        first = collector.get_nowait()
        collector.put(RequestOutput(request_id="r", new_token_ids=[2], new_text="b"))
        second = collector.get_nowait()

        assert first.new_token_ids == [1]
        assert first.new_text == "a"
        assert second.new_token_ids == [2]
        assert second.new_text == "b"

    def test_error_fields_survive_later_clean_output(self):
        collector = RequestOutputCollector(aggregate=True)
        collector.put(
            RequestOutput(
                request_id="r", new_token_ids=[1], error="boom", error_code="X"
            )
        )
        collector.put(RequestOutput(request_id="r", new_token_ids=[2]))

        merged = collector.get_nowait()
        assert merged.error == "boom"
        assert merged.error_code == "X"


class TestAliasRegistryIsolation:
    # M25: one malformed alias used to abort the whole profile load; a corrupt
    # aliases.json only warned and silently emptied the registry.

    @pytest.fixture(autouse=True)
    def _reset_cache(self, monkeypatch):
        import fusion_mlx.model_aliases as ma

        monkeypatch.setattr(ma, "_aliases", None)
        monkeypatch.setattr(ma, "_hf_to_alias", None)
        yield
        ma._aliases = None
        ma._hf_to_alias = None

    def test_bad_alias_does_not_drop_good_ones(self, monkeypatch, caplog):
        import fusion_mlx.model_aliases as ma

        monkeypatch.setattr(
            ma,
            "_load_aliases",
            lambda: {
                "good": "mlx-community/Qwen3.5-9B-4bit",
                "bad": {"modality": "not-a-modality", "hf_path": "x/y"},
            },
        )

        with caplog.at_level(logging.ERROR, logger="fusion_mlx.model_aliases"):
            profiles = ma.list_profiles()

        assert "good" in profiles
        assert "bad" not in profiles
        rejected = [r for r in caplog.records if "rejected" in r.message]
        assert rejected and rejected[0].levelno == logging.ERROR

    def test_corrupt_registry_logs_error(self, monkeypatch, caplog, tmp_path):
        import fusion_mlx.model_aliases as ma

        bad_file = tmp_path / "aliases.json"
        bad_file.write_text("{not json")
        monkeypatch.setattr(ma, "_ALIASES_FILE", bad_file)

        with caplog.at_level(logging.ERROR, logger="fusion_mlx.model_aliases"):
            assert ma._load_aliases() == {}

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors, "corrupt registry must be reported at ERROR"
        assert "aliases.json" in errors[0].message


class TestPytorchFallbackConversion:
    # M26: the .bin fallback double-listed pytorch_model.bin, copied a shard
    # index pointing at files it never wrote, and converted into an existing
    # output directory without asking.

    def _make_model_dir(self, tmp_path):
        torch = pytest.importorskip("torch")
        src = tmp_path / "src"
        src.mkdir()
        torch.save(
            {
                "w": torch.zeros((4, 4), dtype=torch.float16),
                "b": torch.zeros((4,), dtype=torch.float32),
            },
            str(src / "pytorch_model.bin"),
        )
        (src / "config.json").write_text("{}")
        (src / "model.safetensors.index.json").write_text(
            '{"metadata": {}, "weight_map": {"w": "pytorch_model.bin"}}'
        )
        return src

    def test_fallback_skips_stale_index_and_keeps_dtype(self, tmp_path):
        from safetensors import safe_open

        from fusion_mlx.cli_convert import _convert_pytorch_to_safetensors

        src = self._make_model_dir(tmp_path)
        out = _convert_pytorch_to_safetensors(src)

        assert (out / "config.json").exists()
        assert not (out / "model.safetensors.index.json").exists()
        assert not list(out.glob("pytorch_model*.bin"))

        shards = sorted(out.glob("*.safetensors"))
        assert shards
        dtypes = {}
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118 - safe_open is not iterable
                    dtypes[key] = f.get_slice(key).get_dtype()
        assert dtypes["w"] == "F16"
        assert dtypes["b"] == "F32"

    def _run_convert_command(self, monkeypatch, out_dir, force):
        from types import SimpleNamespace

        import fusion_mlx.cli_convert as cc

        calls = []
        monkeypatch.setattr(cc, "_run_convert", lambda model, **kw: calls.append(kw))
        args = SimpleNamespace(
            model="org/model",
            out=str(out_dir),
            quant_bits=None,
            quant_mode="affine",
            quant_group_size=64,
            dtype=None,
            upload_repo=None,
            dequantize=False,
            trust_remote_code=False,
            force=force,
        )
        return cc.convert_command(args), calls

    def test_refuses_existing_output_dir(self, monkeypatch, tmp_path, capsys):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "stale.safetensors").write_text("x")

        rc, calls = self._run_convert_command(monkeypatch, out_dir, force=False)

        assert rc == 1
        assert calls == []
        assert "--force" in capsys.readouterr().err

    def test_force_overwrites(self, monkeypatch, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "stale.safetensors").write_text("x")

        rc, calls = self._run_convert_command(monkeypatch, out_dir, force=True)

        assert rc == 0
        assert len(calls) == 1

    def test_empty_output_dir_needs_no_force(self, monkeypatch, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        rc, calls = self._run_convert_command(monkeypatch, out_dir, force=False)

        assert rc == 0
        assert len(calls) == 1


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
