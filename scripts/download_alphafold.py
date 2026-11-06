"""Fetch AlphaFold models for the proteins behind the ProteinGym assays.

The RBD study shows that structural descriptors of a position are what let a model say anything
about a substitution there without having assayed it. Testing that across ProteinGym needs a
structure per protein, and almost none of these have a solved complex, so predicted models are
used. Each is matched to the assay's target sequence by exact substring search; the offset is
recorded so mutation positions map onto residues, and assays whose target cannot be located in
the model are reported rather than silently aligned.

EBI refuses requests without a User-Agent, so one is sent. It names this client and nothing else.
"""
import json, re, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import proteingym as PG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "alphafold"
UA = "evoclassnet-research/1.0"
API = "https://alphafold.ebi.ac.uk/api/prediction/"
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
ACC_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout).read()


def resolve_accession(uniprot_id, cache):
    """ProteinGym writes UniProt entry names. For some the prefix is already an accession
    (A0A140D2T1_ZIKV); for others it is a gene name (MAFG_HUMAN), which AlphaFold will not accept.
    Resolve the entry name through UniProt in that case."""
    if uniprot_id in cache:
        return cache[uniprot_id]
    head = uniprot_id.split("_")[0]
    acc = head if ACC_RE.match(head) else None
    if acc is None:
        q = urllib.parse.urlencode({"query": f"id:{uniprot_id}", "fields": "accession",
                                    "format": "json", "size": "1"})
        try:
            r = json.loads(get(f"{UNIPROT}?{q}", timeout=40))
            res = r.get("results") or []
            acc = res[0]["primaryAccession"] if res else None
        except Exception:
            acc = None
        time.sleep(0.2)
    cache[uniprot_id] = acc
    return acc


def fetch_one(acc):
    pdb = OUT / f"AF-{acc}.pdb"
    if pdb.exists() and pdb.stat().st_size > 1000:
        return "cached", None
    try:
        meta = json.loads(get(API + acc, timeout=40))
    except urllib.error.HTTPError as e:
        return "no_model", f"api {e.code}"
    except Exception as e:
        return "error", f"api {type(e).__name__}"
    if not meta:
        return "no_model", "empty"
    url = meta[0].get("pdbUrl")
    if not url:
        return "no_model", "no pdbUrl"
    try:
        pdb.write_bytes(get(url))
    except Exception as e:
        return "error", f"pdb {type(e).__name__}"
    (OUT / f"AF-{acc}.json").write_text(json.dumps(
        {k: meta[0].get(k) for k in ("uniprotAccession", "uniprotSequence", "uniprotStart",
                                     "uniprotEnd", "globalMetricValue", "latestVersion",
                                     "modelEntityId")}, indent=1))
    return "ok", None


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ref = PG.reference()
    cpath = OUT / "accessions.json"
    cache = json.loads(cpath.read_text()) if cpath.exists() else {}
    accs_col = []
    for uid in ref.uniprot_id:
        a = resolve_accession(uid, cache)
        accs_col.append(a)
    cpath.write_text(json.dumps(cache, indent=1))
    ref = ref.assign(acc=accs_col)
    unresolved = int(sum(not isinstance(a, str) or not a for a in ref.acc))
    accs = sorted({a for a in ref.acc if isinstance(a, str) and a})
    log(f"{len(accs)} UniProt accessions resolved across {len(ref)} assays "
        f"({unresolved} entry names could not be resolved)")

    status = {}
    for i, acc in enumerate(accs, 1):
        st, why = fetch_one(acc)
        status[acc] = st
        if st not in ("ok", "cached"):
            log(f"[{i}/{len(accs)}] {acc}: {st} ({why})")
        elif i % 20 == 0:
            log(f"[{i}/{len(accs)}] ...")
        time.sleep(0.15)

    # map each assay's target sequence onto its model
    cover = []
    for r in ref.itertuples():
        acc = r.acc if isinstance(r.acc, str) and r.acc else None
        j = OUT / f"AF-{acc}.json" if acc else None
        if j is None or not j.exists():
            cover.append(dict(dms_id=r.dms_id, acc=acc,
                              status="unresolved_id" if not acc else status.get(acc, "missing"),
                              offset=None)); continue
        m = json.loads(j.read_text())
        useq = m.get("uniprotSequence") or ""
        off = useq.find(r.target_seq)
        cover.append(dict(dms_id=r.dms_id, acc=acc,
                          status="matched" if off >= 0 else "seq_mismatch",
                          offset=off if off >= 0 else None,
                          plddt=m.get("globalMetricValue"),
                          model_len=len(useq), target_len=len(r.target_seq)))
    json.dump(cover, open(OUT / "coverage.json", "w"), indent=1)
    n_ok = sum(1 for c in cover if c["status"] == "matched")
    log(f"models downloaded for {sum(1 for v in status.values() if v in ('ok','cached'))}"
        f"/{len(accs)} accessions")
    log(f"assays whose target sequence maps onto its model: {n_ok}/{len(cover)}")
    from collections import Counter
    log("status: " + ", ".join(f"{k}={v}" for k, v in Counter(c["status"] for c in cover).items()))
