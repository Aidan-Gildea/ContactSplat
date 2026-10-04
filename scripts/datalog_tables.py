#!/usr/bin/env python3
"""Print the datalogger's results tables as markdown, straight from the result files. Never hand-edit the output."""
import json, glob, os, statistics as st
import sys
O = sys.argv[1] if len(sys.argv) > 1 else "output"   # the repo's output/ folder
# Usage: python scripts/datalog_tables.py [output_dir] > tables.md
RECT = "camera-rgb-rectified-1008-h1512"
SCENES = ["Outside_20260812_141244", "Hall_20260929_213101", "Room_20260929_211650"] + [f"park{a}_{b}" for a in (3, 4, 7) for b in range(3)]
ENV = {"Outside_20260812_141244": "outdoor", "Hall_20260929_213101": "indoor", "Room_20260929_211650": "indoor"}
WEARER = {"Outside_20260812_141244": "A (6'1\")", "Hall_20260929_213101": "A (6'1\")", "Room_20260929_211650": "A (6'1\")"}
H = {"Outside_20260812_141244": 1.667, "Hall_20260929_213101": 1.667, "Room_20260929_211650": 1.667, "park3_0": 1.588, "park3_1": 1.583,
     "park3_2": 1.586, "park4_0": 1.615, "park4_1": 1.600, "park4_2": 1.596, "park7_0": 1.590, "park7_1": 1.593, "park7_2": 1.589}
short = lambda s: s.split("_2026")[0]
def mean(v): v = list(v.values()) if isinstance(v, dict) else v; v = [x for x in v if isinstance(x, (int, float))]; return st.mean(v) if v else None
print("### Recordings\n\n| recording | environment | wearer | posed frames | glasses-to-floor (measured) |\n|---|---|---|---|---|")
for s in SCENES:
    cams = f"{O}/{s}/{RECT}/cameras.json"
    n = (lambda j: len(j["train"]) + len(j["test"]))(json.load(open(cams))) if os.path.exists(cams) else "—"   # valid == test split
    print(f"| {short(s)} | {ENV.get(s,'outdoor')} | {WEARER.get(s,'B (5′10″)')} | {n} | {H[s]:.3f} m |")
print("\n### Splat quality (held-out frames)\n\n| recording | variant | Gaussian cap | PSNR | SSIM | LPIPS | test frames |\n|---|---|---|---|---|---|---|")
for s in SCENES:
    for var, sub in (("RS on", RECT), ("RS off", RECT + "-rsoff")):
        f = f"{O}/{s}/{sub}/test_logs.json"
        if not os.path.exists(f): continue
        j = json.load(open(f)); cap = "1.0 M" if (s.startswith("Room") and var == "RS on") else "1.5 M"
        print(f"| {short(s)} | {var} | {cap} | {mean(j['psnr']):.2f} | {mean(j['ssim']):.3f} | {mean(j['lpips']):.3f} | {len(j['psnr'])} |")
print("\n### Collision meshes (scored with each recording's measured glasses-to-floor height)\n")
print("| recording | method | triangles | floor coverage | largest hole (m²) | floor height error, median (cm) |\n|---|---|---|---|---|---|")
for s in SCENES:
    tag = "h%d" % round(H[s] * 1000)
    for kind, label in (("photogrammetry", "MVS"), ("mps-mesh", "MPS points")):
        for pre, suf in (("report", ""), ("report_decimated", " · decimated")):
            f = f"{O}/{kind}/{s}/{RECT}/{pre}_{tag}.json"
            if not os.path.exists(f):
                f0 = f"{O}/{kind}/{s}/{RECT}/{pre}.json"
                if not (os.path.exists(f0) and abs(json.load(open(f0)).get("eye_height_m", 0) - H[s]) < 0.002): continue
                f = f0
            j = json.load(open(f)); fl = j["floor"]; e = fl["height_error_covered_cm"]["median"]
            print(f"| {short(s)} | {label}{suf} | {j['hygiene']['triangles']:,} | {fl['coverage']:.3f} | {fl['largest_hole_m2']:.2f} | {e:.1f} |")
