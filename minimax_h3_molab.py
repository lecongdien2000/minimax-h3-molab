"""Interactive MiniMax H3 notebook for marimo molab.

Model weights stay on Hugging Face; embedding them as Base64 would enlarge this
file and cannot make the model fit in memory. GPU use is interactive only.
"""

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # MiniMax H3 on molab

    Generate a video **with synchronized audio** from text or a first-frame image.
    This runs open H3 weights in your molab GPU session. Attach the **96 GB GPU**
    using the notebook specs button before starting. The session ends after at
    most 12 hours or 90 minutes idle; this is an interactive notebook, not an API.

    The transformer is a community Q4 GGUF conversion and the MiniMax H3 text
    encoder is a community INT8 safetensors conversion. Quantization saves
    memory but may change quality compared with
    the official BF16 checkpoint. Files download on the first run; nothing is
    Base64-embedded in the notebook.

    [Model and license](https://huggingface.co/MiniMaxAI/MiniMax-H3) ·
    [SGLang H3 guide](https://docs.sglang.ai/cookbook/diffusion/MiniMax/MiniMax-H3/)
    """)
    return (mo,)


@app.cell
def _(mo):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    work = Path("/tmp/minimax-h3-molab")
    work.mkdir(parents=True, exist_ok=True)
    gpu_mib = 0
    try:
        _gpu_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        _gpu_name, _gpu_memory = _gpu_query.stdout.splitlines()[0].rsplit(",", 1)
        gpu_mib = int(_gpu_memory.strip())
        gpu_info = f"{_gpu_name.strip()}, {gpu_mib / 1024:.1f} GiB VRAM"
    except Exception as exc:
        gpu_info = f"NVIDIA GPU not visible to this session ({exc})"
    disk = shutil.disk_usage(work)
    mo.md(f"**GPU:** {gpu_info}  \n**Free temporary disk:** {disk.free / 2**30:.1f} GiB  \n**Python:** {sys.version.split()[0]}")
    return Path, os, shutil, subprocess, sys, work, gpu_mib, disk


@app.cell
def _(mo):
    install_button = mo.ui.run_button(label="Install SGLang Diffusion", kind="success")
    install_button
    return (install_button,)


@app.cell
def _(install_button, mo, subprocess, sys, work):
    mo.stop(not install_button.value)
    venv = work / "sglang-venv"
    _cli = venv / "bin" / "sglang"
    commands = [
        [sys.executable, "-m", "pip", "install", "-q", "uv"],
        [sys.executable, "-m", "uv", "venv", "--python", "3.12", str(venv)],
        [sys.executable, "-m", "uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--no-cache", "sglang[diffusion]==0.5.23", "comfy-kitchen==0.2.41", "ninja==1.13.2", "--prerelease=allow"],
    ]
    for _command in commands:
        _result = subprocess.run(_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if _result.returncode:
            raise RuntimeError(f"Install failed ({_result.returncode}):\n{_result.stdout[-6000:]}")
    if not _cli.exists():
        raise RuntimeError("Installation finished, but the SGLang command was not created.")
    mo.md("SGLang installed in a separate Python 3.12 environment. Generate below without restarting the notebook.")
    return


@app.cell
def _(mo):
    prompt = mo.ui.text_area(
        label="Describe the video and soundtrack",
        value="A red fox runs through snowy pines, cinematic camera movement, snow crunching underfoot.",
        rows=4,
    )
    first_frame = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg", ".webp"], kind="area", label="Optional first frame")
    short_edge = mo.ui.dropdown(options=["480", "544", "768"], value="480", label="Short edge (16:9)")
    duration = mo.ui.dropdown(options=["5", "7", "10"], value="5", label="Duration (seconds)")
    steps = mo.ui.slider(start=20, stop=50, step=5, value=40, label="Sampling steps")
    seed = mo.ui.number(value=42, start=0, step=1, label="Seed")
    mo.vstack([prompt, first_frame, mo.hstack([short_edge, duration]), mo.hstack([steps, seed])])
    return prompt, first_frame, short_edge, duration, steps, seed


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Generate H3 video", kind="success")
    run_button
    return (run_button,)


@app.cell
def _(Path, duration, first_frame, mo, os, prompt, run_button, seed, short_edge, shutil, steps, subprocess, sys, work):
    mo.stop(not run_button.value)
    import json
    _cli = work / "sglang-venv" / "bin" / "sglang"
    if not _cli.exists():
        raise RuntimeError("Install SGLang with the button above first.")
    _venv_python = work / "sglang-venv" / "bin" / "python"
    _gpu_check = subprocess.run(
        [str(_venv_python), "-c", "import torch; print(torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)"],
        capture_output=True, text=True,
    )
    if _gpu_check.returncode:
        raise RuntimeError("SGLang's Python could not check CUDA:\n" + _gpu_check.stderr[-2000:])
    _gpu_bytes = int(_gpu_check.stdout.strip())
    if _gpu_bytes == 0:
        raise RuntimeError("No NVIDIA GPU is visible to SGLang. Attach the GPU in Molab notebook specs and wait for the session to restart, then retry.")
    if _gpu_bytes < 80 * 2**30:
        raise RuntimeError(f"SGLang sees only {_gpu_bytes / 2**30:.1f} GiB VRAM. Select Molab's 96 GB GPU in notebook specs.")
    old_encoder_cache = work / "hf-cache" / "hub" / "models--DeepBeepMeep--MiniMax-H3"
    if old_encoder_cache.is_dir():
        shutil.rmtree(old_encoder_cache)
    if shutil.disk_usage(work).free < 60 * 2**30:
        raise RuntimeError("At least 60 GiB of temporary disk is needed for quantized weights and caches. This molab session has too little free disk.")
    if not prompt.value.strip():
        raise ValueError("Enter a text prompt.")
    _kitchen_check = subprocess.run([str(_venv_python), "-c", "import comfy_kitchen"], capture_output=True, text=True)
    if _kitchen_check.returncode:
        _kitchen_install = subprocess.run(
            [sys.executable, "-m", "uv", "pip", "install", "--python", str(_venv_python), "--no-cache", "comfy-kitchen==0.2.41"],
            capture_output=True, text=True,
        )
        if _kitchen_install.returncode:
            raise RuntimeError("Could not install comfy-kitchen in SGLang's Python environment:\n" + _kitchen_install.stderr[-4000:])
    _venv_bin = work / "sglang-venv" / "bin"
    _ninja = _venv_bin / "ninja"
    if not _ninja.exists():
        _ninja_install = subprocess.run(
            [sys.executable, "-m", "uv", "pip", "install", "--python", str(_venv_python), "--no-cache", "ninja==1.13.2"],
            capture_output=True, text=True,
        )
        if _ninja_install.returncode or not _ninja.exists():
            raise RuntimeError("Could not install Ninja for SGLang's JIT kernels:\n" + _ninja_install.stderr[-4000:])
    _nvcc = shutil.which("nvcc")
    if not _nvcc:
        raise RuntimeError("CUDA nvcc is missing from Molab. Attach the NVIDIA GPU and restart the notebook session.")
    import re
    _nvcc_version = subprocess.run([_nvcc, "--version"], capture_output=True, text=True)
    _release = re.search(r"release (\d+)\.(\d+)", _nvcc_version.stdout)
    if _nvcc_version.returncode or not _release or _release.group(1) != "13":
        raise RuntimeError("Expected CUDA 13 nvcc; found:\n" + (_nvcc_version.stdout + _nvcc_version.stderr)[-2000:])
    _cuda_minor = int(_release.group(2))
    _cuda_headers = work / "cuda-headers"
    _cuda_include = _cuda_headers / "nvidia" / "cu13" / "include"
    if not (_cuda_include / "nv" / "target").is_file():
        _cccl_install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade", "--target", str(_cuda_headers),
             f"nvidia-cuda-cccl>=13.{_cuda_minor},<13.{_cuda_minor + 1}"],
            capture_output=True, text=True,
        )
        if _cccl_install.returncode or not (_cuda_include / "nv" / "target").is_file():
            raise RuntimeError("Could not install CUDA CCCL headers for nvcc:\n" + _cccl_install.stderr[-4000:])
    _cuda_runtime = work / "cuda-runtime"
    _cuda_runtime_lib = _cuda_runtime / "nvidia" / "cu13" / "lib"
    _cudart = _cuda_runtime_lib / "libcudart.so.13"
    if not _cudart.is_file():
        _runtime_install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade", "--target", str(_cuda_runtime),
             f"nvidia-cuda-runtime>=13.{_cuda_minor},<13.{_cuda_minor + 1}"],
            capture_output=True, text=True,
        )
        if _runtime_install.returncode or not _cudart.is_file():
            raise RuntimeError("Could not install the CUDA runtime library for SGLang's JIT linker:\n" + _runtime_install.stderr[-4000:])
    _cuda_link = work / "cuda-link"
    _cuda_link.mkdir(exist_ok=True)
    _cudart_link = _cuda_link / "libcudart.so"
    if _cudart_link.is_symlink() or _cudart_link.exists():
        _cudart_link.unlink()
    _cudart_link.symlink_to(_cudart)
    _toolchain = _venv_bin.parent / "lib" / "python3.12" / "site-packages" / "sglang" / "kernels" / "jit" / "utils" / "compile" / "toolchain.py"
    _toolchain_source = _toolchain.read_text(encoding="utf-8")
    _link_flags = 'return flags + [f"-L{cuda_home()}/lib64", "-lcudart"]'
    _fixed_link_flags = f'return flags + [f"-L{{cuda_home()}}/lib64", "-L{_cuda_link}", "-lcudart"]'
    if _link_flags in _toolchain_source:
        _toolchain.write_text(_toolchain_source.replace(_link_flags, _fixed_link_flags), encoding="utf-8")
    elif _fixed_link_flags not in _toolchain_source:
        raise RuntimeError("SGLang's CUDA link flags have changed; cannot add the CUDA runtime library safely.")

    output = work / "h3-output.mp4"
    if output.exists():
        output.unlink()
    request = {
        "task": "t2va",
        "conditions": [],
        "target": {
            "short_edge": int(short_edge.value),
            "aspect_ratio": "16:9",
            "duration_seconds": float(duration.value),
        },
    }
    if first_frame.value:
        upload = first_frame.value[0]
        suffix = Path(upload.name).suffix.lower()
        image_path = work / f"first-frame{suffix}"
        image_path.write_bytes(upload.contents)
        request["task"] = "fl2va"
        request["conditions"] = [{
            "type": "image",
            "uri": image_path.as_uri(),
            "role": "keyframe",
            "frame_index": 0,
        }]
        request["target"]["aspect_ratio"] = "auto"
    request_path = work / "h3-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    command = [
        str(_cli), "generate",
        "--config", str(request_path),
        "--model-path", "MiniMaxAI/MiniMax-H3",
        "--model-variant", "fl2va",
        "--component-weights-paths.transformer", "leejet/MiniMax-H3-GGUF/minimax_h3_fl2va-Q4_K_M.gguf",
        "--component-weights-paths.text_encoder", "Comfy-Org/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "--num-gpus", "1",
        "--attention-backend", "fa",
        "--performance-mode", "memory",
        "--enable-torch-compile", "false",
        "--prompt", prompt.value.strip(),
        "--num-inference-steps", str(steps.value),
        "--seed", str(int(seed.value)),
        "--save-output",
        "--output-path", str(work),
        "--output-file-name", output.name,
    ]
    env = os.environ.copy()
    env["HF_HOME"] = str(work / "hf-cache")
    env["PATH"] = str(_venv_bin) + os.pathsep + env.get("PATH", "")
    env["NVCC_PREPEND_FLAGS"] = (
        f"-I{_cuda_include} -I{_cuda_include / 'cccl'} " + env.get("NVCC_PREPEND_FLAGS", "")
    ).strip()
    env["LIBRARY_PATH"] = str(_cuda_link) + os.pathsep + env.get("LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(_cuda_runtime_lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    with (work / "generation.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode or not output.exists():
        tail = (work / "generation.log").read_text(encoding="utf-8", errors="replace")[-8000:]
        reason = f"SGLang exited {result.returncode}" if result.returncode else "SGLang produced no video despite exiting 0"
        raise RuntimeError(f"H3 generation failed: {reason}. Log tail:\n{tail}")
    mo.vstack([mo.md(f"Generated **{output.name}**"), mo.video(str(output))])
    return


if __name__ == "__main__":
    app.run()
