"""Генератор сырых данных об оттоке клиентов.

Почему синтетика, а не готовый датасет:
  * пайплайн воспроизводится офлайн — не зависит от доступности внешнего URL;
  * на занятии 15 нужен управляемый дрейф, а в статичном датасете его негде взять.

Флаг --drift выдаёт «данные следующего квартала»: тариф подорожал, доля
оптики выросла, клиенты стали чаще звонить в поддержку. Модель, обученная
на исходном распределении, на таких данных деградирует — это и разбирается
на занятии про мониторинг.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_params, resolve
from src.logging_setup import setup_logging

log = setup_logging()

CONTRACTS = ["month-to-month", "one_year", "two_year"]
INTERNET = ["fiber", "dsl", "none"]
PAYMENTS = ["electronic_check", "mailed_check", "bank_transfer", "credit_card"]


def generate(n: int, seed: int, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if drift:
        contract_p = [0.62, 0.24, 0.14]      # больше клиентов без обязательств
        internet_p = [0.58, 0.30, 0.12]      # оптика вытеснила DSL
        charges_shift, calls_shift = 12.0, 0.6
    else:
        contract_p = [0.50, 0.29, 0.21]
        internet_p = [0.42, 0.42, 0.16]
        charges_shift, calls_shift = 0.0, 0.0

    tenure = rng.integers(1, 73, size=n)
    contract = rng.choice(CONTRACTS, size=n, p=contract_p)
    internet = rng.choice(INTERNET, size=n, p=internet_p)
    payment = rng.choice(PAYMENTS, size=n, p=[0.36, 0.22, 0.21, 0.21])

    base_charge = np.where(internet == "fiber", 78.0, np.where(internet == "dsl", 52.0, 21.0))
    monthly = np.clip(rng.normal(base_charge + charges_shift, 9.0), 15.0, 180.0)
    total = np.round(monthly * tenure * rng.uniform(0.92, 1.05, size=n), 2)

    gb = np.clip(rng.gamma(shape=2.2, scale=14.0, size=n), 0.5, 300.0)
    support_calls = rng.poisson(1.1 + calls_shift, size=n)
    tech_support = (rng.random(n) < 0.34).astype(int)
    senior = (rng.random(n) < 0.16).astype(int)

    # Логит оттока: короткий стаж, помесячный контракт, дорогой тариф
    # и звонки в поддержку толкают клиента к уходу.
    logit = (
        -1.15
        - 0.045 * tenure
        + 0.020 * (monthly - 60.0)
        + 0.34 * support_calls
        - 0.55 * tech_support
        + 0.28 * senior
        + np.where(contract == "month-to-month", 1.05, np.where(contract == "one_year", 0.15, -0.55))
        + np.where(internet == "fiber", 0.42, np.where(internet == "none", -0.38, 0.0))
        + np.where(payment == "electronic_check", 0.36, 0.0)
        + rng.normal(0, 0.45, size=n)
    )
    churn = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:07d}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charges": np.round(monthly, 2),
            "total_charges": total,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "num_support_calls": support_calls,
            "has_tech_support": tech_support,
            "avg_monthly_gb": np.round(gb, 2),
            "is_senior": senior,
            "churn": churn,
        }
    )

    # Реальные выгрузки не бывают чистыми: оставляем пропуски в total_charges
    # у новых клиентов — на занятии 3 их придётся обработать осознанно.
    missing_idx = rng.choice(n, size=max(1, n // 100), replace=False)
    df.loc[missing_idx, "total_charges"] = np.nan
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Сгенерировать сырой датасет оттока")
    parser.add_argument("--drift", action="store_true", help="распределение 'следующего квартала'")
    parser.add_argument("--out", default=None, help="куда писать CSV")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    params = load_params()
    n = args.n or params["data"]["n_samples"]
    seed = params["seed"] + (1000 if args.drift else 0)
    out = Path(args.out) if args.out else resolve(params["data"]["raw_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate(n=n, seed=seed, drift=args.drift)
    df.to_csv(out, index=False)
    log.info("написано %s строк в %s (drift=%s, churn rate=%.3f)",
             len(df), out, args.drift, df["churn"].mean())


if __name__ == "__main__":
    main()
