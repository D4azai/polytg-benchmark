from polytg.training.callbacks import EarlyStopping


def test_early_stopping_resets_only_on_material_improvement() -> None:
    callback = EarlyStopping(patience=2, min_delta=0.1)
    assert not callback.update(10.0)
    assert not callback.update(9.95)
    assert callback.update(9.94)
    assert callback.epochs_without_improvement == 2
    assert not callback.update(9.8)
    assert callback.best_value == 9.8

