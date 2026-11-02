"""
comfyui-pe-llama-cpp
--------------------
ComfyUI node to use an external LLM on llama.cpp for fast prompt enhancement.
Invokes an external llama.cpp executable (llama.exe / llama-cli / llama-completion) via the command line  - passing a prompt, system prompt, and parameters - and reads the stdout.

ComfyUI node: prompt enhancer for text to image generation.
"""

import os
import re
import shlex
import subprocess
import tempfile
import time

DEFAULT_EXE = r"path/to/your/llama.exe"
DEFAULT_MODEL = r"path/to/your/pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF/pe_t2i_heretic-Q4_K_M.gguf"

_env_cache = {}

def _fmt_cmd(cmd):
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)

def _build_env(setvars_path):
    """Runs oneAPI’s setvars.bat once and captures the resulting environment."""
    env = os.environ.copy()
    if not setvars_path:
        return env
    if setvars_path not in _env_cache:
        if os.name == "nt":
            args = ["cmd", "/c", "call", setvars_path, ">nul", "2>&1", "&&", "set"]
        else:
            args = ["bash", "-c", 'source "$1" >/dev/null 2>&1 && env', "_", setvars_path]
        res = subprocess.run(
            args,
            capture_output=True, text=True, errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        captured = {}
        for line in res.stdout.splitlines():
            if "=" in line and not line.startswith("="):
                k, v = line.split("=", 1)
                captured[k] = v
        _env_cache[setvars_path] = captured
    env.update(_env_cache[setvars_path])
    return env


def _clean_chat_output(text):
    """Best-effort cleanup of the output from the chat mode of llama-cli."""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^\[\s*Prompt:.*\]$", s) or s in ("Exiting...", "> EOF by user"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


class LlamaCppCLIEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exe_path": ("STRING", {"default": DEFAULT_EXE}),
                "model_path": ("STRING", {"default": DEFAULT_MODEL}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["completion (Manual ChatML)", "chat (--jinja, single-turn)"],),
                "thinking": ("BOOLEAN", {"default": False}),
                "max_tokens": ("INT", {"default": 400, "min": 16, "max": 8192}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 131072}),
                "n_gpu_layers": ("INT", {"default": 99, "min": -1, "max": 999}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 1.5, "min": -2.0, "max": 3.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "flash_attn": (["auto", "on", "off", "do_not_pass"],),
                "timeout_s": ("INT", {"default": 600, "min": 10, "max": 7200}),
            },
            "optional": {
                "subcommand": (["auto", "completion", "cli", "none"],),
                "extra_args": ("STRING", {"default": "", "multiline": False}),
                "oneapi_setvars": ("STRING", {"default": ""}),
                "bypass": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "thinking", "command")
    FUNCTION = "run"
    CATEGORY = "text/llama.cpp"

    def run(self, exe_path, model_path, system_prompt, prompt, mode, thinking,
            max_tokens, n_ctx, n_gpu_layers, temperature, top_k, top_p, min_p,
            repeat_penalty, presence_penalty, seed, flash_attn, timeout_s,
            subcommand="auto", extra_args="", oneapi_setvars="", bypass=False):
        
        if bypass:
            print("[LlamaCppCLI] Bypass enabled: prompt enhancer skipped.")
            return (prompt, "", "bypass")
            
        if not os.path.isfile(exe_path):
            raise FileNotFoundError(f"Executable not found: {exe_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        chat_mode = mode.startswith("chat")
        tmp_files = []

        def write_tmp(content):
            f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8", newline="\n")
            f.write(content)
            f.close()
            tmp_files.append(f.name)
            return f.name

        try:
            # Recent builds of llama.cpp feature a single "llama" binary with subcommands (llama cli / llama completion / llama server ...).
            # The classic binaries (llama-cli.exe, llama-completion.exe), by contrast, do not take subcommands.
            if subcommand == "auto":
                is_unified = os.path.splitext(os.path.basename(exe_path))[0].lower() == "llama"
                sub = ["cli" if chat_mode else "completion"] if is_unified else []
            elif subcommand == "none":
                sub = []
            else:
                sub = [subcommand]

            cmd = [exe_path] + sub + ["-m", model_path,
                   "-c", str(n_ctx), "-n", str(max_tokens), "-ngl", str(n_gpu_layers),
                   "--temp", str(temperature), "--top-k", str(top_k), "--top-p", str(top_p),
                   "--min-p", str(min_p), "--repeat-penalty", str(repeat_penalty),
                   "--presence-penalty", str(presence_penalty),
                   "-s", str(seed % (2 ** 32)),
                   "--no-display-prompt"]

            if flash_attn != "do_not_pass":
                cmd += ["-fa", flash_attn]

            if chat_mode:
                cmd += ["--jinja", "-st", "--simple-io"]
                if system_prompt.strip():
                    cmd += ["-sysf", write_tmp(system_prompt)]
                cmd += ["-f", write_tmp(prompt)]
                if not thinking:
                    cmd += ["--chat-template-kwargs", '{"enable_thinking":false}']
            else:
                full = ""
                if system_prompt.strip():
                    full += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                full += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                full += "<think>\n" if thinking else "<think>\n\n</think>\n\n"
                cmd += ["-no-cnv", "-f", write_tmp(full)]

            if extra_args.strip():
                cmd += [a.strip('"') for a in shlex.split(extra_args, posix=False)]

            print("[LlamaCppCLI] " + _fmt_cmd(cmd))
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
        print(f"[LlamaCppCLI] Total time (including model loading): {elapsed:.1f}s")
        if stats:
            print("[LlamaCppCLI] llama.cpp statistics:\n  " + "\n  ".join(stats))
        else:
            print("[LlamaCppCLI] No statistics found in stderr.")

        if proc.returncode != 0 and not out.strip():
            tail = "\n".join(err.strip().splitlines()[-25:])
            raise RuntimeError(f"llama.cpp exited with code {proc.returncode}:\n{tail}")

        text = _clean_chat_output(out) if chat_mode else out.strip()
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
            print("[LlamaCppCLI] WARNING: end-of-thinking marker not found, max_tokens probably too low.")
            think_txt, text = text, ""

        return (text, think_txt, _fmt_cmd(cmd))


NODE_CLASS_MAPPINGS = {"LlamaCppCLIEnhancer": LlamaCppCLIEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"LlamaCppCLIEnhancer": "Llama.cpp CLI Prompt Enhancer"}
