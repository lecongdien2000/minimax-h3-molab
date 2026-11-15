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

    The transformer and text encoder below are community Q4 GGUF conversions of
    MiniMax H3. Quantization saves memory but may change quality compared with
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
    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        gpu_info = f"{gpu.name}, {gpu.total_memory / 2**30:.1f} GiB VRAM" if gpu else "No NVIDIA GPU attached"
    except Exception as exc:
        gpu_info = f"GPU check failed: {exc}"
    disk = shutil.disk_usage(work)
    mo.md(f"**GPU:** {gpu_info}  \n**Free temporary disk:** {disk.free / 2**30:.1f} GiB  \n**Python:** {sys.version.split()[0]}")
    return Path, os, shutil, subprocess, sys, work, gpu, disk


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
        [sys.executable, "-m", "uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--no-cache", "sglang[diffusion]==0.5.23", "--prerelease=allow"],
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
    width = mo.ui.dropdown(options=["864", "960", "1344"], value="864", label="Width")
    height = mo.ui.dropdown(options=["480", "544", "768"], value="480", label="Height")
    frames = mo.ui.dropdown(options=["101", "125", "173"], value="125", label="Frames at 24 fps")
    steps = mo.ui.slider(start=20, stop=50, step=5, value=40, label="Sampling steps")
    seed = mo.ui.number(value=42, start=0, step=1, label="Seed")
    mo.vstack([prompt, first_frame, mo.hstack([width, height, frames]), mo.hstack([steps, seed])])
    return prompt, first_frame, width, height, frames, steps, seed


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Generate H3 video", kind="success")
    run_button
    return (run_button,)


@app.cell
def _(Path, disk, first_frame, frames, gpu, height, mo, os, prompt, run_button, seed, shutil, steps, subprocess, width, work):
    mo.stop(not run_button.value)
    if gpu is None or gpu.total_memory < 80 * 2**30:
        raise RuntimeError("Attach the 96 GB NVIDIA GPU in molab notebook specs, then retry.")
    if disk.free < 60 * 2**30:
        raise RuntimeError("At least 60 GiB of temporary disk is needed for quantized weights and caches. This molab session has too little free disk.")
    if not prompt.value.strip():
        raise ValueError("Enter a text prompt.")
    _cli = work / "sglang-venv" / "bin" / "sglang"
    if not _cli.exists():
        raise RuntimeError("Install SGLang with the button above first.")

    output = work / "h3-output.mp4"
    if output.exists():
        output.unlink()
    command = [
        str(_cli), "generate",
        "--model-path", "MiniMaxAI/MiniMax-H3",
        "--model-variant", "fl2va",
        "--component-weights-paths.transformer", "leejet/MiniMax-H3-GGUF/minimax_h3_fl2va-Q4_K_M.gguf",
        "--component-weights-paths.text_encoder", "DeepBeepMeep/MiniMax-H3/Qwen3-VL-32B-Instruct/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf",
        "--num-gpus", "1",
        "--attention-backend", "fa",
        "--performance-mode", "memory",
        "--enable-torch-compile", "false",
        "--prompt", prompt.value.strip(),
        "--width", width.value,
        "--height", height.value,
        "--num-frames", frames.value,
        "--fps", "24",
        "--num-inference-steps", str(steps.value),
        "--seed", str(int(seed.value)),
        "--save-output",
        "--output-path", str(work),
        "--output-file-name", output.name,
    ]
    if first_frame.value:
        upload = first_frame.value[0]
        suffix = Path(upload.name).suffix.lower()
        image_path = work / f"first-frame{suffix}"
        image_path.write_bytes(upload.contents)
        command.extend(["--image-path", str(image_path)])

    env = os.environ.copy()
    env["HF_HOME"] = str(work / "hf-cache")
    with (work / "generation.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode or not output.exists():
        tail = (work / "generation.log").read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"H3 generation failed (exit {result.returncode}). Log tail:\n{tail}")
    mo.vstack([mo.md(f"Generated **{output.name}**"), mo.video(str(output))])
    return


if __name__ == "__main__":
    app.run()
