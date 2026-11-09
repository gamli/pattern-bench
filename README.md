# PatternBench

Stitch overlapping scans of sewing patterns into a single image.

## Download

Download `pattern-bench.exe` from the [latest release](https://github.com/gamli/pattern-bench/releases/latest).
The Windows x64 executable includes Python and its dependencies.

## Build and publish a Windows version

1. Push the source changes you want to release.
2. Open [Actions → Build Windows EXE](https://github.com/gamli/pattern-bench/actions/workflows/windows-release.yml).
3. Click **Run workflow**, select the branch, and enter a new version such as `0.1.0` (without `v`).
4. Click **Run workflow**.

GitHub builds the single-file EXE on Windows using `package-distribution.ps1`
and the dependencies in `uv.lock`. It checks that the packaged executable
starts and can apply calibration with a fresh user configuration. After these
checks pass, it creates the tag `v0.1.0` and publishes a GitHub Release with
`pattern-bench.exe` as a directly downloadable asset. Existing versions are
not overwritten.

Pull requests that change the application or build configuration also build
and check the EXE, without publishing a release.

## Build locally on Windows

With [uv](https://docs.astral.sh/uv/) installed, run this in the project directory:

```powershell
.\package-distribution.ps1
```

The script installs missing project dependencies and writes `dist/pattern-bench.exe`.
