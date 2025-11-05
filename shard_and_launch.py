#!/usr/bin/env python3
import argparse
import os
import pathlib
import shutil
import shlex
import subprocess
import time

def list_models(root):
    exts = {".glb", ".gltf", ".fbx"}
    # exts = {".glb", ".gltf", ".obj", ".fbx", ".blend"}
    root = pathlib.Path(root)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def make_shards(models, shards_root, num_shards):
    """
    Split the model list into N shards (one per GPU) and COPY the files
    into each shard so Blender can write cache/sidecar data beside them.
    """
    shards = []
    shards_root = pathlib.Path(shards_root)
    shards_root.mkdir(parents=True, exist_ok=True)
    
    # remove existing shard folders
    for i in range(num_shards):
        d = shards_root / f"shard_{i}"
        if d.exists():
            shutil.rmtree(d)

    # create shard folders
    for i in range(num_shards):
        d = shards_root / f"shard_{i}"
        d.mkdir(exist_ok=True)
        shards.append(d)

    # round-robin assign models to shards and COPY them
    for idx, m in enumerate(models):
        target = shards[idx % num_shards] / m.name
        if not target.exists():
            try:
                shutil.copy2(m, target)
            except Exception as e:
                # print(f"⚠️  Could not copy {m} → {target}: {e}")
                pass
    return shards

def main():
    time_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",   required=True)
    ap.add_argument("--output_dir",  required=True)
    # ap.add_argument("--results_dir", required=True)
    ap.add_argument("--blender",     required=True)  # /home/.../blender
    ap.add_argument("--pipeline",    required=True)  # /labworking/.../3Dpipeline_1.py
    ap.add_argument("--gpus",        required=True, help="comma list, e.g. 0,1 or 0")
    ap.add_argument("--shards_root", default=str(pathlib.Path.home()/ "vgl" / "shards"))
    ap.add_argument("--limit",       type=int, default=-1, help="cap number of models for this run")
    args = ap.parse_args()

    models = list_models(args.input_dir)
    if args.limit > 0:
        models = models[:args.limit]
    if not models:
        print("No models found.")
        return

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    shards = make_shards(models, args.shards_root, len(gpu_ids))

    procs = []
    logs_root = pathlib.Path(args.shards_root) / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    
    os.makedirs(args.output_dir, exist_ok=True)

    for gpu, shard_dir in zip(gpu_ids, shards):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        # env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        # env["__NV_PRIME_RENDER_OFFLOAD_PROVIDER"] = "nvidia"
        env["INPUT_DIR"]   = str(shard_dir)
        env["OUTPUT_DIR"]  = args.output_dir
        env["RESULTS_DIR"] = os.path.join(args.output_dir, 'logs')
        env["BLENDER_PATH"] = args.blender
        # VGL_ENV_CITY / VGL_ENV_NIGHT (if set) will be inherited automatically

        log_path = logs_root / f"gpu{gpu}.log"
        cmd = [
            args.blender,
            "--background",
            "--python", args.pipeline
        ]
        print("Launching:", " ".join(shlex.quote(c) for c in cmd),
              "on GPU", gpu, "input:", shard_dir)
        f = open(log_path, "w")
        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=args.output_dir,   # run from a writable place
        )
        procs.append((p, f, log_path))

    # wait & close logs
    exit_codes = []
    for p, f, log_path in procs:
        rc = p.wait()
        f.close()
        print(f"[GPU log] {log_path} → exit {rc}")
        exit_codes.append(rc)

    bad = [rc for rc in exit_codes if rc != 0]
    if bad:
        raise SystemExit(f"{len(bad)} shard(s) failed; check logs under {logs_root}")
    time_end = time.time()
    print(f"All shards completed successfully in {time_end - time_start:.2f} seconds.")

if __name__ == "__main__":
    main()

