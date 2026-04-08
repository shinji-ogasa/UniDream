# Optimization Status

最適化ループの現在地は次で書き出せる。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\write_optimization_status.ps1
```

出力:

- `checkpoints/optimization_status.json`
- `checkpoints/optimization_status.md`

判定は単純で、

- `pending`: まだ output directory が無い
- `ready`: directory はあるが結果ファイルはまだ無い
- `done`: 結果ファイルが入っている

issue ごとの current state を機械的に残す用途に使う。
