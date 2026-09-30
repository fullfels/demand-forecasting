from forecast import FEATURES, build_features, make_sales_series, train_and_evaluate


def test_feature_builder_has_no_missing_values():
    featured = build_features(make_sales_series(120))
    assert featured[FEATURES].isna().sum().sum() == 0
    assert featured["date"].is_monotonic_increasing


def test_forecast_beats_simple_scale_threshold():
    _, metrics, result = train_and_evaluate()
    assert len(result) == 90
    assert metrics["mape_percent"] < 20
