import json, glob
BASE="/workspace/eval/results"
# rate-0.5 comparison: baseline, attention top-k, unicomp, and UniAttn at 3 lambdas
CFGS=[("nopruning","no pruning"),("atte0.5","atte 0.5"),("unic0.5b","unic 0.5"),
      ("uniattn0.5_l0","uniattn l0"),("uniattn0.5_l03","uniattn l.3"),("uniattn0.5_l07","uniattn l.7")]
def scores(cfg):
    out={}
    for jf in glob.glob(f"{BASE}/{cfg}/**/*_results.json", recursive=True):
        try: d=json.load(open(jf))
        except: continue
        for task,metrics in d.get("results",{}).items():
            for mk,mv in metrics.items():
                if isinstance(mv,(int,float)):
                    out[(task, mk.split(",")[0])]=mv
    return out
data={c:scores(c) for c,_ in CFGS}
ROWS=[
 ("POPE acc","pope","pope_accuracy",1),
 ("POPE F1","pope","pope_f1_score",1),
 ("TextVQA","textvqa_val","exact_match",1),
 ("ScienceQA","scienceqa_img","exact_match",1),
 ("MMMU","mmmu_val","mmmu_acc",1),
 ("MME-perc","mme","mme_perception_score",None),
 ("MME-cog","mme","mme_cognition_score",None),
]
hdr=["benchmark"]+[n for _,n in CFGS]
print(" | ".join(hdr)); print("-|-".join("-"*len(h) for h in hdr))
def cell(v,scale):
    if v is None: return "-"
    return f"{v*100:.1f}" if scale==1 else f"{v:.0f}"
tab=[]
for label,task,metric,scale in ROWS:
    row=[label]+[cell(data[c].get((task,metric)),scale) for c,_ in CFGS]
    tab.append((label,task,metric,scale)); print(" | ".join(row))

print("\n=== retention relative to no pruning (%) ===")
print(" | ".join(hdr)); print("-|-".join("-"*len(h) for h in hdr))
for label,task,metric,scale in tab:
    base=data["nopruning"].get((task,metric))
    if not base: continue
    row=[label]+[ (f"{data[c].get((task,metric))/base*100:.1f}%" if data[c].get((task,metric)) is not None else "-") for c,_ in CFGS]
    print(" | ".join(row))

print("\n=== 4-metric accuracy mean (POPE-acc, TextVQA, ScienceQA, MMMU) ===")
acc=[("pope","pope_accuracy"),("textvqa_val","exact_match"),("scienceqa_img","exact_match"),("mmmu_val","mmmu_acc")]
for c,n in CFGS:
    vals=[data[c].get(k) for k in acc]; vals=[v for v in vals if v is not None]
    if vals: print(f"  {n:10s}: {sum(vals)/len(vals)*100:.2f}  (n={len(vals)})")
