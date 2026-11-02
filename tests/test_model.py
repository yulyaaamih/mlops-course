from src.data.prepare import fill_missing as clean


def test_more_support_calls_increase_churn_risk(trained_pipeline, raw_df, cols):
    """Направленный тест: больше обращений в поддержку -> выше риск ухода.
    Нарушение означает перепутанные признаки или таргет."""
    base = clean(raw_df)[cols].head(200).copy()
    calm, angry = base.copy(), base.copy()
    calm.loc[:, "num_support_calls"] = 0
    angry.loc[:, "num_support_calls"] = 8

    assert (
        trained_pipeline.predict_proba(angry)[:, 1].mean() > trained_pipeline.predict_proba(calm)[:, 1].mean()
    )


def test_probabilities_are_valid(trained_pipeline, raw_df, cols):
    proba = trained_pipeline.predict_proba(clean(raw_df).tail(400)[cols])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba.std() > 0.01, "модель выдаёт почти константу — признак сломанного обучения"


def test_prediction_is_row_independent(trained_pipeline, raw_df, cols):
    """Предсказание для строки не зависит от соседей в батче.
    Иначе /predict и /predict/batch дадут разные ответы для одного клиента."""
    X = clean(raw_df).tail(400)[cols]
    assert (
        abs(
            trained_pipeline.predict_proba(X.head(1))[:, 1][0]
            - trained_pipeline.predict_proba(X.head(50))[:, 1][0]
        )
        < 1e-9
    )


def test_long_contract_decreases_churn_risk(trained_pipeline, raw_df, cols):
    base = clean(raw_df)[cols].head(200).copy()
    monthly, two_year = base.copy(), base.copy()
    monthly.loc[:, "contract_type"] = "month-to-month"
    two_year.loc[:, "contract_type"] = "two_year"

    risk_monthly = trained_pipeline.predict_proba(monthly)[:, 1].mean()
    risk_two_year = trained_pipeline.predict_proba(two_year)[:, 1].mean()
    assert risk_two_year < risk_monthly, (
        f"two_year {risk_two_year:.3f} не ниже month-to-month {risk_monthly:.3f}"
    )
