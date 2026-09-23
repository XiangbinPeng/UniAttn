import json, glob
BASE="/workspace/eval/results"
CFGS=[("nopruning","no pruning"),("atte0.5","atte0.5"),("unic0.5b","unic0.5"),
      ("uniattn0.5_l03","uniattn0.5 λ.3"),("uniattn0.8_l03","uniattn0.8 λ.3"),("uniattn0.9_l03","uniattn0.9 λ.3")]
def scores(cfg):
    out={}
    for jf in glob.glob(f"{BASE}/{cfg}/**/*_results.json", recursive=True):
        try: d=json.load(open(jf))
        except: continue
        for task,metrics in d.get("results",{}).items():
            for mk,mv in metrics.items():
                if isinstance(mv,(int,float)): out[(task, mk.split(",")[0])]=mv
    return out
data={c:scores(c) for c,_ in CFGS}
ROWS=[("POPE acc","pope","pope_accuracy",1),("POPE F1","pope","pope_f1_score",1),
 ("TextVQA","textvqa_val","exact_match",1),("ScienceQA","scienceqa_img","exact_match",1),
 ("MMMU","mmmu_val","mmmu_acc",1),("MME-perc","mme","mme_perception_score",None),
 ("MME-cog","mme","mme_cognition_score",None)]
hdr=["benchmark"]+[n for _,n in CFGS]
def cell(v,s): return "-" if v is None else (f"{v*100:.1f}" if s==1 else f"{v:.0f}")
print("## Raw scores\n")
print("| "+" | ".join(hdr)+" |"); print("|"+"---|"*len(hdr))
for label,task,metric,scale in ROWS:
    print("| "+" | ".join([label]+[cell(data[c].get((task,metric)),scale) for c,_ in CFGS])+" |")
print("\n## Retention relative to no pruning (%)\n")
print("| "+" | ".join(hdr)+" |"); print("|"+"---|"*len(hdr))
for label,task,metric,scale in ROWS:
    base=data["nopruning"].get((task,metric))
    if not base: continue
    row=[label]+[(f"{data[c].get((task,metric))/base*100:.1f}%" if data[c].get((task,metric)) is not None else "-") for c,_ in CFGS]
    print("| "+" | ".join(row)+" |")
print("\n## 4-metric accuracy mean (POPE-acc, TextVQA, ScienceQA, MMMU)\n")
acc=[("pope","pope_accuracy"),("textvqa_val","exact_match"),("scienceqa_img","exact_match"),("mmmu_val","mmmu_acc")]
print("| config | mean | vs atte0.5 |"); print("|---|---|---|")
means={}
for c,n in CFGS:
    vals=[data[c].get(k) for k in acc]; vals=[v for v in vals if v is not None]
    if vals: means[c]=sum(vals)/len(vals)*100
base_a=means.get("atte0.5")
for c,n in CFGS:
    if c in means:
        d=means[c]-base_a if base_a else 0
        print(f"| {n} | {means[c]:.2f} | {d:+.2f} |")
