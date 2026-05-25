#!/usr/bin/env python3
"""Create a small QUIC-like temporal drift dataset for smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_rows(days: int, services: list[str], flows_per_service_day: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2025-01-01")
    asns = ["asn_a", "asn_b", "asn_c", "asn_d"]
    countries = ["US", "DE", "CZ", "JP"]

    for day in range(days):
        ts = base + pd.Timedelta(days=day)
        for service_i, service in enumerate(services):
            for _ in range(flows_per_service_day):
                drift_cert = service == "svc_alpha" and day >= 12
                drift_provider = service == "svc_beta" and day >= 16
                drift_network = service == "svc_gamma" and 9 <= day <= 14
                drift_protocol = service == "svc_delta" and day >= 18

                provider_idx = (service_i + int(drift_provider) * 2) % len(asns)
                country_idx = (service_i + int(drift_provider) * 2) % len(countries)

                rows.append(
                    {
                        "time": ts + pd.Timedelta(minutes=int(rng.integers(0, 1440))),
                        "service": service,
                        "QUIC_VERSION": "v2" if drift_protocol else "v1",
                        "QUIC_TOKEN_LENGTH": int(rng.normal(80 if drift_protocol else 36, 8)),
                        "QUIC_TLS_EXT_LEN": int(rng.normal(280 if drift_cert else 160, 20)),
                        "QUIC_PACKETS": int(rng.normal(8 if drift_cert else 6, 1)),
                        "DST_ASN": asns[provider_idx],
                        "DST_COUNTRY": countries[country_idx],
                        "DURATION": float(rng.gamma(2.0, 0.6 if drift_network else 0.25)),
                        "BYTES": float(rng.normal(4800 if drift_network else 3200, 450)),
                        "PPI_PKT_0": float(rng.normal(1200 if drift_cert else 750, 100)),
                        "PPI_PKT_1": float(rng.normal(850 if drift_protocol else 680, 80)),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--flows-per-service-day", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_rows(
        days=args.days,
        services=["svc_alpha", "svc_beta", "svc_gamma", "svc_delta", "svc_epsilon"],
        flows_per_service_day=args.flows_per_service_day,
        seed=args.seed,
    )
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows to {out}")


if __name__ == "__main__":
    main()
