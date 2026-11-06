"""Fetch coronavirus genomes from NCBI for the sequence-classification task.

The original draft claimed a five-way classification of 48,891 genomes but no such experiment
existed. This builds a real, smaller version of that task, designed so it carries the same
methodological point as the deep-mutational-scanning study: genomes are labelled by genus, and
the evaluation can be split either at random or by *species*, so that no species appears in both
training and test. A random split over coronavirus genomes is close to meaningless, because
near-identical sequences from the same outbreak land on both sides of it.

SARS-CoV-2 is excluded from the Betacoronavirus class. It has over three million deposited
genomes against a few hundred for whole genera, and including it would make the class a measure
of one outbreak's sequencing effort rather than of the genus.
"""
import json, os, sys, time, urllib.parse, urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "genomes"
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
EMAIL = os.environ.get("NCBI_EMAIL", "")
GENERA = ["Alphacoronavirus", "Betacoronavirus", "Gammacoronavirus", "Deltacoronavirus"]
MAX_PER_SPECIES = 12          # keeps one heavily sequenced species from dominating its genus
TARGET_PER_GENUS = 420
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def _get(tool, **kw):
    kw.setdefault("tool", "evoclassnet")
    if EMAIL:
        kw["email"] = EMAIL
    url = BASE + tool + ".fcgi?" + urllib.parse.urlencode(kw)
    for attempt in range(5):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def search(genus, retmax=1800):
    term = f'"{genus}"[Organism] AND complete genome[Title] AND 25000:35000[SLEN]'
    if genus == "Betacoronavirus":
        term += ' NOT "Severe acute respiratory syndrome coronavirus 2"[Organism]'
    r = json.loads(_get("esearch", db="nuccore", term=term, retmax=retmax, retmode="json"))
    return r["esearchresult"]["idlist"]


def summaries(ids, chunk=200):
    out = {}
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        r = json.loads(_get("esummary", db="nuccore", id=",".join(part), retmode="json"))
        for uid in r["result"].get("uids", []):
            d = r["result"][uid]
            out[uid] = dict(accession=d.get("accessionversion", uid), organism=d.get("organism", "?"),
                            title=d.get("title", ""), length=int(d.get("slen", 0)))
        time.sleep(0.4)
        log(f"  summaries {min(i + chunk, len(ids))}/{len(ids)}")
    return out


def pick(summ, target, max_per_species):
    """Species-balanced subset: at most `max_per_species` per organism, breadth before depth."""
    by = defaultdict(list)
    for uid, d in summ.items():
        by[d["organism"]].append(uid)
    chosen, depth = [], 0
    while len(chosen) < target and depth < max_per_species:
        added = 0
        for org in sorted(by):
            if len(by[org]) > depth and len(chosen) < target:
                chosen.append(by[org][depth]); added += 1
        if not added:
            break
        depth += 1
    return chosen


def fetch_fasta(ids, chunk=50):
    parts = []
    for i in range(0, len(ids), chunk):
        parts.append(_get("efetch", db="nuccore", id=",".join(ids[i:i + chunk]),
                          rettype="fasta", retmode="text").decode("utf8", "replace"))
        time.sleep(0.4)
        log(f"  fasta {min(i + chunk, len(ids))}/{len(ids)}")
    return "".join(parts)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for genus in GENERA:
        fa = OUT / f"{genus}.fasta"
        mj = OUT / f"{genus}.json"
        if fa.exists() and mj.exists():
            log(f"{genus}: cached")
            meta[genus] = json.load(open(mj))
            continue
        log(f"{genus}: searching")
        ids = search(genus)
        log(f"{genus}: {len(ids)} hits; fetching summaries")
        summ = summaries(ids)
        sel = pick(summ, TARGET_PER_GENUS, MAX_PER_SPECIES)
        log(f"{genus}: selected {len(sel)} records across "
            f"{len({summ[u]['organism'] for u in sel})} species")
        fa.write_text(fetch_fasta(sel))
        rec = {summ[u]["accession"]: summ[u] for u in sel}
        json.dump(rec, open(mj, "w"), indent=1)
        meta[genus] = rec
    for g, r in meta.items():
        log(f"{g}: {len(r)} genomes, {len({v['organism'] for v in r.values()})} species")
