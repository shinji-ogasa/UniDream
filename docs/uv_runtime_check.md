# uv Runtime Check

`uv run` が詰まる時は、次で runtime 状態をまとめて確認できる。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_uv_runtime.ps1
```

既定では workspace 内の `.uv-cache` を `UV_CACHE_DIR` に使う。

出力:

- `checkpoints/uv_runtime_check.txt`

確認するもの:

- `uv --version`
- `uv python find`
- `uv run python -V`
- `uv run python -c "import sys; print(sys.executable)"`
- `uv run --python C:\Users\Sophie\anaconda3\envs\UniDream\python.exe python -V`
