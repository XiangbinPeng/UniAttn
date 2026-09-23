import os, signal, subprocess, sys, time, urllib.request
# usage: restart_enc_uniattn.py RATE STRATEGY [UNIATTN_LAMBDA]
RATE=sys.argv[1]; STRAT=sys.argv[2] if len(sys.argv)>2 else ""
LAM=sys.argv[3] if len(sys.argv)>3 else ""
PORTS=[30000+i for i in range(8)]
out=subprocess.run(["ps","-eo","pid,cmd"],capture_output=True,text=True).stdout
procs={}
for ln in out.splitlines():
    if "launch_server --encoder-only" not in ln: continue
    pid=int(ln.split()[0])
    for p in PORTS:
        if f"--port {p}" in ln: procs[p]=pid
if not procs: sys.exit("no encoders found")
src=next(iter(procs.values()))
env={}
for kv in open(f"/proc/{src}/environ","rb").read().split(b"\0"):
    if kv and b"=" in kv:
        k,v=kv.decode(errors="ignore").split("=",1); env[k]=v
env["CUDA_VISIBLE_DEVICES"]="0"
print(f"env snapshot: {len(env)} vars from pid={src}")
for p,pid in sorted(procs.items()):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
print(f"stopped {len(procs)} encoders")
for _ in range(40):
    time.sleep(1)
    if not any(os.path.exists(f"/proc/{pid}") for pid in procs.values()): break
time.sleep(3)
for i,p in enumerate(PORTS):
    cmd=["python3","-u","-m","sglang.launch_server","--encoder-only","--enable-multimodal",
         "--model-path","/path/to/model","--host","0.0.0.0","--port",str(p),
         "--tp-size","1","--dist-init-addr",f"127.0.0.1:{47000+i}","--enable-metrics",
         "--trust-remote-code","--mm-attention-backend","fa3",
         "--mm-process-config",'{"image": {"max_pixels": 262144}}',
         "--keep-mm-feature-on-device","--quantization","w4afp8",
         "--vision-quantization","fp8","--image-pruning-rate",RATE,
         "--encoder-transfer-backend","mooncake",
         "--mooncake-ib-device","mlx5_bond_6","--disaggregation-ib-device","mlx5_bond_6"]
    if STRAT: cmd += ["--image-pruning-strategy", STRAT]
    if LAM:   cmd += ["--uniattn-lambda", LAM]
    log=open(f"/workspace/epd_test/logs/enc_{p}.log","w")
    subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(f"relaunched 8 encoders rate={RATE} strategy={STRAT or '(default)'} lambda={LAM or '(default)'}")
deadline=time.time()+480
while time.time()<deadline:
    ok=0
    for p in PORTS:
        try: urllib.request.urlopen(f"http://127.0.0.1:{p}/health",timeout=2); ok+=1
        except Exception: pass
    if ok==8: print(">>> all 8 healthy"); break
    time.sleep(10)
else: print(f">>> TIMEOUT, only {ok}/8 healthy"); sys.exit(1)
