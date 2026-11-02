"""Extract the Karamanlis & Gollisch (2021) session archives.

The dataset is a git-annex repository (GIN): a .rar that was not fetched is a
~64-byte text pointer "/annex/objects/MD5-s<size>--<md5>". With --download,
such archives are fetched from GIN into --download-dir and checked against the
pointer's size and MD5. Original files are never modified.

Usage:
    python scripts/extract.py --data-dir data --out-dir extracted [--download]
"""
import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

GIN_URL = ("https://gin.g-node.org/gollischlab/"
           "Karamanlis_Gollisch_2021_RGC_spiketrains_natural_image_encoding/raw/master/")
POINTER_RE = re.compile(rb"^/annex/objects/MD5-s(\d+)--([0-9a-f]{32})\s*$")


def annex_pointer(path):
    """Return (size, md5) if `path` is a git-annex pointer, else None."""
    if path.stat().st_size > 1024:
        return None
    m = POINTER_RE.match(path.read_bytes())
    return (int(m.group(1)), m.group(2).decode()) if m else None


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, size, md5, download_dir):
    dest = download_dir / name
    if dest.exists() and dest.stat().st_size == size and md5sum(dest) == md5:
        return dest
    download_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"  downloading {name} ({size / 1e6:.1f} MB)...", flush=True)
    with urllib.request.urlopen(GIN_URL + name) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if tmp.stat().st_size != size or md5sum(tmp) != md5:
        tmp.unlink()
        raise RuntimeError(f"{name}: size/MD5 mismatch after download")
    tmp.rename(dest)
    return dest


def find_extractor():
    for tool, args in [("unrar", ["x", "-o+", "-idq"]), ("unar", ["-f", "-q", "-o"]),
                       ("7z", ["x", "-y", "-bd"])]:
        if shutil.which(tool):
            return tool, args
    sys.exit("No RAR extractor found: install unrar, unar or 7z (p7zip).")


def extract(archive, out, tool, args):
    """Extract into out/<archive_name>/. The archives contain a top-level
    folder named like the archive, which is flattened away."""
    target = out / archive.stem
    tmp = out / f".tmp_{archive.stem}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if tool == "unrar":
        cmd = [tool, *args, str(archive), str(tmp) + "/"]
    elif tool == "unar":
        cmd = [tool, *args, str(tmp), str(archive)]
    else:
        cmd = [tool, *args, f"-o{tmp}", str(archive)]
    subprocess.run(cmd, check=True)
    inner = tmp / archive.stem
    src = inner if inner.is_dir() and len(list(tmp.iterdir())) == 1 else tmp
    shutil.rmtree(target, ignore_errors=True)
    src.rename(target)
    shutil.rmtree(tmp, ignore_errors=True)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("extracted"))
    p.add_argument("--download", action="store_true", help="fetch git-annex pointers from GIN")
    p.add_argument("--download-dir", type=Path, default=None,
                   help="where fetched archives go (default: <data-dir>/rar_full)")
    a = p.parse_args()
    download_dir = a.download_dir or a.data_dir / "rar_full"
    tool, args = find_extractor()
    print(f"Extractor: {tool}")

    ok, failed = [], []
    for rar in sorted(a.data_dir.glob("*.rar")):
        name = rar.name
        try:
            ptr = annex_pointer(rar)
            if ptr:
                if not a.download:
                    raise RuntimeError("git-annex pointer (content not fetched); rerun with --download")
                rar = fetch(name, *ptr, download_dir)
            target = extract(rar, a.out_dir, tool, args)
            ok.append(name)
            print(f"[ok]   {name} -> {target}")
        except Exception as e:  # report and continue with the other sessions
            failed.append((name, str(e)))
            print(f"[FAIL] {name}: {e}")

    print(f"\n{len(ok)} extracted, {len(failed)} failed")
    for name, err in failed:
        print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
