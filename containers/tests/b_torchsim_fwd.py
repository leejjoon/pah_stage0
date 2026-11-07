import time, sys, torch; sys.argv=["x"]; exec(open("/t/b_torchsim_pilot.py").read().split("raw = mace_off")[0])
from mace.cli.convert_e3nn_cueq import run as to_cueq
raw = mace_off(model="small", default_dtype="float32", device="cuda", return_raw_model=True)
for tag, r in [("e3nn", raw), ("cueq", to_cueq(raw, device="cuda"))]:
    m = MaceTorchSimModel(model=r, device=dev, dtype=torch.float32, compute_forces=True, compute_stress=False)
    s = ts.io.atoms_to_state(batch, device=dev, dtype=torch.float32)
    for _ in range(3): out = m(s)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): out = m(s)
    torch.cuda.synchronize(); print(f"{tag}: forward {(time.perf_counter()-t)/20*1e3:.1f} ms for {s.n_atoms} atoms")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p: m(s); torch.cuda.synchronize()
    print(p.key_averages().table(sort_by="cpu_time_total", row_limit=6, max_name_column_width=50))
