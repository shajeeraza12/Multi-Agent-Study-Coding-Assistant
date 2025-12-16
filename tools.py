import subprocess
import tempfile
import textwrap
import os
from typing import Literal

Language = Literal["python", "c", "cpp"]


def run_code(code: str, language: Language = "python", timeout: int = 5) -> str:
    code = textwrap.dedent(code).strip()
    if not code:
        return "No code provided."

    try:
        if language == "python":
            return _run_python(code, timeout=timeout)
        elif language == "c":
            return _run_c_or_cpp(code, lang="c", timeout=timeout)
        elif language == "cpp":
            return _run_c_or_cpp(code, lang="cpp", timeout=timeout)
        else:
            return f"Unsupported language: {language}"
    except subprocess.TimeoutExpired:
        return "Execution timed out."
    except Exception as e:
        return f"Execution error: {e}"


def _run_python(code: str, timeout: int) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name

    try:
        result = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.returncode != 0:
            return f"Python error (exit {result.returncode}):\n{err or out}"
        return out or "(no output)"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _run_c_or_cpp(code: str, lang: str, timeout: int) -> str:
    suffix = ".c" if lang == "c" else ".cpp"
    compiler = "gcc" if lang == "c" else "g++"

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, f"program{suffix}")
        bin_path = os.path.join(tmpdir, "program")

        with open(src_path, "w") as f:
            f.write(code)

        compile_proc = subprocess.run(
            [compiler, src_path, "-o", bin_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if compile_proc.returncode != 0:
            err = compile_proc.stderr.strip() or compile_proc.stdout.strip()
            return f"Compilation failed:\n{err}"

        run_proc = subprocess.run(
            [bin_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = run_proc.stdout.strip()
        err = run_proc.stderr.strip()
        if run_proc.returncode != 0:
            return f"Program exited with code {run_proc.returncode}:\n{err or out}"
        return out or "(no output)"
