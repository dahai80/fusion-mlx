from fusion_mlx.scheduler.monkeypatches import (
    _bump_uid_row_epoch,
    _realigned_rows,
    _register_uid_rows,
    _unregister_uid_rows_for_model,
)


class _FakeModel:
    pass


def _make_rows(model, uid, sampler, lps):
    _register_uid_rows(model, [uid], [sampler], [lps])


def test_same_epoch_entry_realigns_row():
    model = _FakeModel()
    sampler = object()
    lps = [["p1"]]
    _make_rows(model, 7, sampler, lps)
    samplers, out_lps, drift = _realigned_rows(model, [7], [None], [[]])
    assert samplers == [sampler]
    assert out_lps == [lps]
    assert drift


def test_stale_epoch_entry_ignored():
    model = _FakeModel()
    dead_sampler = object()
    _make_rows(model, 7, dead_sampler, [["dead"]])
    # Generator discarded: epoch bump invalidates every registration
    # made before it, even though the entries remain in the dict.
    _bump_uid_row_epoch()
    cur_sampler = object()
    cur_lps = [["live"]]
    samplers, out_lps, drift = _realigned_rows(model, [7], [cur_sampler], cur_lps)
    # Recycled uid must keep its current row state, not the dead entry.
    assert samplers == [cur_sampler]
    assert out_lps == [cur_lps[0]]
    assert not drift


def test_re_registration_after_epoch_bump_is_visible():
    model = _FakeModel()
    _make_rows(model, 3, object(), [["old"]])
    _bump_uid_row_epoch()
    new_sampler = object()
    new_lps = [["new"]]
    _make_rows(model, 3, new_sampler, new_lps)
    samplers, out_lps, drift = _realigned_rows(model, [3], [None], [[]])
    assert samplers == [new_sampler]
    assert out_lps == [new_lps]
    assert drift


def test_unregister_all_clears_entries_regardless_of_epoch():
    model = _FakeModel()
    _make_rows(model, 3, object(), [[]])
    _unregister_uid_rows_for_model(model)
    _bump_uid_row_epoch()
    samplers, out_lps, drift = _realigned_rows(model, [3], [None], [[]])
    assert samplers == [None]
    assert out_lps == [[]]
    assert not drift
