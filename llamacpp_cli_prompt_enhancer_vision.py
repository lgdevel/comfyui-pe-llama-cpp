"""
comfyui-pe-llama-cpp
--------------------
ComfyUI node to use an external LLM on llama.cpp for fast prompt enhancement.
Invokes an external llama.cpp executable (llama.exe / llama-cli / llama-completion) via the command line, passing a prompt, system prompt, and parameters, and reads the stdout.

ComfyUI node: prompt enhancer for image edit.
This node also accepts one or more images (IMAGE) and passes them to llama.cpp (using the --mmproj and --image flags) along with the text.
It must be placed in the same folder as llamacpp_cli_prompt_enhancer.py (as it reuses its functions).
"""
import os
import re
import shlex
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from .llamacpp_cli_prompt_enhancer import (
    DEFAULT_EXE, _build_env, _clean_chat_output, _fmt_cmd,
)

DEFAULT_MODEL_VISION = r"path/to/your/pottokao/Qwen-Image-2.1-PE-I2I-Heretic-GGUF/pe_i2i_heretic-Q4_K_M.gguf"

def _find_mmproj(model_path):
    """Look for an *mmproj*.gguf file in the model folder."""
    folder = os.path.dirname(model_path)
    cands = [f for f in os.listdir(folder)
             if "mmproj" in f.lower() and f.lower().endswith(".gguf")]
    base = re.split(r"[-_.](q\d|iq\d|f16|f32|bf16)", os.path.basename(model_path).lower())[0]
    matching = [f for f in cands if base in f.lower()]
    if matching:
        return os.path.join(folder, matching[0])
    if len(cands) == 1:
        return os.path.join(folder, cands[0])
    return None


class LlamaCppCLIEnhancerVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exe_path": ("STRING", {"default": DEFAULT_EXE}),
                "model_path": ("STRING", {"default": DEFAULT_MODEL_VISION}),
                "mmproj_path": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_mode": (["added to prompt", "flag -sysf"],),
                "thinking": ("BOOLEAN", {"default": False}),
                "max_tokens": ("INT", {"default": 600, "min": 16, "max": 16384}),
                "n_ctx": ("INT", {"default": 32768, "min": 512, "max": 131072}),
                "max_image_side": ("INT", {"default": 1024, "min": 0, "max": 4096}),
                "n_gpu_layers": ("INT", {"default": 99, "min": -1, "max": 999}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 1.5, "min": -2.0, "max": 3.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "flash_attn": (["auto", "on", "off", "do_not_pass"],),
                "timeout_s": ("INT", {"default": 900, "min": 10, "max": 7200}),
            },
            "optional": {
                "subcommand": ("STRING", {"default": "cli"}),
                "extra_args": ("STRING", {"default": "", "multiline": False}),
                "oneapi_setvars": ("STRING", {"default": ""}),
                "bypass": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "thinking", "command")
    FUNCTION = "run"
    CATEGORY = "text/llama.cpp"

    def run(self, image, exe_path, model_path, mmproj_path, system_prompt, prompt,
            system_mode, thinking, max_tokens, n_ctx, max_image_side, n_gpu_layers,
            temperature, top_k, top_p, min_p, repeat_penalty, presence_penalty, seed,
            flash_attn, timeout_s, subcommand="auto", extra_args="",
            oneapi_setvars="", bypass=False):

        if bypass:
            print("[LlamaCppCLI-Vision] Bypass enabled: prompt enhancer skipped.")
            return (prompt, "", "bypass")

        if not os.path.isfile(exe_path):
            raise FileNotFoundError(f"Executable not found: {exe_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        mmproj = mmproj_path.strip() or _find_mmproj(model_path)
        if not mmproj or not os.path.isfile(mmproj):
            raise FileNotFoundError(
                "mmproj file not found. Specify the path in mmproj_path "
                "(the *mmproj*.gguf file corresponding to the model is required).")

        tmp_files = []

        def new_tmp(suffix):
            f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            f.close()
            tmp_files.append(f.name)
            return f.name

        def write_text(content):
            p = new_tmp(".txt")
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)
            return p

        try:
            # Save each image from the batch as a temporary PNG (optionally resized)
            img_paths = []
            for i in range(image.shape[0]):
                arr = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                im = Image.fromarray(arr[..., :3])
                if max_image_side and max(im.size) > max_image_side:
                    im.thumbnail((max_image_side, max_image_side), Image.LANCZOS)
                p = new_tmp(".png")
                im.save(p)
                img_paths.append(p)

            sc = subcommand.strip()
            sub = [] if sc.lower() in ("", "none") else (["cli"] if sc.lower() == "auto" else sc.split())
            
            cmd = [exe_path] + sub + ["-m", model_path, "--mmproj", mmproj,
                   "-c", str(n_ctx), "-n", str(max_tokens), "-ngl", str(n_gpu_layers),
                   "--temp", str(temperature), "--top-k", str(top_k), "--top-p", str(top_p),
                   "--min-p", str(min_p), "--repeat-penalty", str(repeat_penalty),
                   "--presence-penalty", str(presence_penalty),
                   "-s", str(seed % (2 ** 32)), "--jinja"]

            if flash_attn != "do_not_pass":
                cmd += ["-fa", flash_attn]
            if not thinking:
                cmd += ["--chat-template-kwargs", '{"enable_thinking":false}']

            if system_prompt.strip() and system_mode == "flag -sysf":
                cmd += ["-sysf", write_text(system_prompt)]
                user_text = prompt
            elif system_prompt.strip():
                user_text = f"{system_prompt}\n\n{prompt}"
            else:
                user_text = prompt
            cmd += ["-f", write_text(user_text)]

            for p in img_paths:
                cmd += ["--image", p]

            if extra_args.strip():
                cmd += [a.strip('"') for a in shlex.split(extra_args, posix=False)]

            print("[LlamaCppCLI-Vision] " + _fmt_cmd(cmd))
            t0 = time.time()

            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_build_env(oneapi_setvars.strip()),
                timeout=timeout_s,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"llama.cpp timed out after {timeout_s}s")
        finally:
            for p in tmp_files:
                try:
                    os.remove(p)
                except OSError:
                    pass

        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")

        elapsed = time.time() - t0
        stats = [l.strip() for l in err.splitlines()
                 if re.search(r"perf|eval time|total time|load time|tokens per second", l, re.I)]
        print(f"[LlamaCppCLI-Vision] Images: {len(img_paths)} | Total time: {elapsed:.1f}s")
        if stats:
            print("[LlamaCppCLI-Vision] llama.cpp statistics:\n  " + "\n  ".join(stats))

        if proc.returncode != 0 and not out.strip():
            tail = "\n".join(err.strip().splitlines()[-25:])
            raise RuntimeError(f"llama.cpp exited with code {proc.returncode}:\n{tail}")

        text = _clean_chat_output(out)
        text = re.sub(r"\s*\[end of text\]\s*$", "", text).strip()
        think_txt = ""

        if "[End thinking]" in text:
            think_txt, text = text.split("[End thinking]", 1)
            think_txt = think_txt.replace("[Start thinking]", "").strip()
            text = text.strip()
        elif "</think>" in text:
            think_txt, text = text.split("</think>", 1)
            think_txt = think_txt.replace("<think>", "").strip()
            text = text.strip()
        elif thinking:
            print("[LlamaCppCLI-Vision] WARNING: 'end thinking' marker not found; max_tokens is likely too low.")
            think_txt, text = text, ""
        
        return (text, think_txt, _fmt_cmd(cmd))


NODE_CLASS_MAPPINGS = {"LlamaCppCLIEnhancerVision": LlamaCppCLIEnhancerVision}
NODE_DISPLAY_NAME_MAPPINGS = {"LlamaCppCLIEnhancerVision": "Llama.cpp CLI Prompt Enhancer (Image Edit)"}
