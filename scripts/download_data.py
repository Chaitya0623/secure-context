"""Download and pre-process the public datasets used by the evaluation.

    python3 scripts/download_data.py            # everything (about 400 MB download, 2 to 6 minutes)
    python3 scripts/download_data.py --skip-acs # skip the Census files

Uses only the Python standard library. Raw downloads go to data/raw/, small
processed files to data/processed/ (both ignored by git).

Datasets
  Financial-Advisory-Clients (Hugging Face, MIT)           client profiles
  Bitext Wealth Management (Hugging Face, CDLA-Sharing-1.0) advisor/client request intents
  Gretel synthetic_pii_finance_multilingual (HF, Apache-2.0) financial documents with PII spans
  Federal Reserve Survey of Consumer Finances 2022 (public) household wealth and portfolios
  Census ACS PUMS 2023 1-year (public)                      population directory for re-identification
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

HF = "https://huggingface.co/datasets"
FILES = {
    "advisory_clients.csv": f"{HF}/Finance-Agentic-AI/Financial-Advisory-Clients/resolve/main/financial_advisory_clients.csv",
    "bitext_wealth.csv": f"{HF}/bitext/Bitext-wealth-management-llm-chatbot-training-dataset/resolve/main/"
                         "bitext-wealth-management-llm-chatbot-training-dataset.csv",
    "gretel_en_test.parquet": f"{HF}/gretelai/synthetic_pii_finance_multilingual/resolve/main/data/"
                              "English_test-00000-of-00001.parquet",
    "scfp2022excel.zip": "https://www.federalreserve.gov/econres/files/scfp2022excel.zip",
}
ACS_STATES = ["ny", "nj", "ca", "tx", "fl", "il"]
ACS_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_p{st}.zip"
ACS_COLS = ["ST", "PUMA", "AGEP", "SEX", "MAR", "OCCP", "PINCP", "WAGP", "SCHL", "PWGTP"]
SCF_COLS = ["YY1", "Y1", "WGT", "AGE", "MARRIED", "KIDS", "OCCAT1", "OCCAT2", "EDCL", "INCOME", "NETWORTH",
            "FIN", "LIQ", "CDS", "NMMF", "STOCKS", "BOND", "RETQLIQ", "SAVBND", "CASHLI", "OTHMA", "OTHFIN"]


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name}")
        return True
    print(f"  downloading {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "secure-context-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        print(f"    {dest.stat().st_size / 1e6:.1f} MB")
        return True
    except Exception as e:  # keep going so one failure does not stop the rest
        print(f"    failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def process_scf():
    z = RAW / "scfp2022excel.zip"
    if not z.exists():
        return
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as f, gzip.open(OUT / "scf2022.csv.gz", "wt", newline="") as g:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))
            cols = {c.upper(): c for c in reader.fieldnames}
            keep = [c for c in SCF_COLS if c in cols]
            w = csv.writer(g)
            w.writerow(keep)
            n = 0
            for row in reader:
                w.writerow([row[cols[c]] for c in keep])
                n += 1
    print(f"  scf2022.csv.gz: {n:,} rows, columns {len(keep)}/{len(SCF_COLS)}")


def process_acs(states):
    out = gzip.open(OUT / "acs_persons.csv.gz", "wt", newline="")
    w = csv.writer(out)
    w.writerow(ACS_COLS)
    total = 0
    counts = []
    for st in states:
        z = RAW / f"csv_p{st}.zip"
        if not z.exists():
            continue
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                n = 0
                for row in reader:
                    try:
                        if int(row["AGEP"]) < 25:
                            continue
                    except ValueError:
                        continue
                    row.setdefault("ST", row.get("STATE", ""))  # 2023 files call it STATE
                    if not row["ST"]:
                        row["ST"] = row.get("STATE", "")
                    w.writerow([row.get(c, "") for c in ACS_COLS])
                    n += 1
        print(f"  {st.upper()}: {n:,} adults")
        counts.append(f"{st.upper()},{n}")
        total += n
    out.close()
    (OUT / "acs_state_order.txt").write_text("\n".join(counts) + "\n")
    print(f"  acs_persons.csv.gz: {total:,} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-acs", action="store_true")
    ap.add_argument("--states", nargs="*", default=ACS_STATES)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Hugging Face and Federal Reserve files")
    ok = {k: fetch(u, RAW / k) for k, u in FILES.items()}
    for k in ("advisory_clients.csv", "bitext_wealth.csv", "gretel_en_test.parquet"):
        if ok[k]:
            shutil.copy(RAW / k, OUT / k)
    process_scf()

    if not args.skip_acs:
        print("Census ACS PUMS person files")
        for st in args.states:
            fetch(ACS_URL.format(st=st), RAW / f"csv_p{st}.zip")
        process_acs(args.states)

    print("\nProcessed files:")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  {p.stat().st_size / 1e6:.1f} MB")
    missing = [k for k, v in ok.items() if not v]
    if missing:
        print("Some downloads failed:", ", ".join(missing), "(rerun the script to retry)")
        sys.exit(1)


if __name__ == "__main__":
    main()
