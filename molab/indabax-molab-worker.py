import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    token = mo.ui.text(kind="password", label="Kaggle API token", full_width=True)
    start = mo.ui.run_button(label="Start training")
    mo.vstack([mo.md("Paste your Kaggle API token, then click **Start training**. The token stays in this session: it is never saved in the notebook."), token, start])
    return start, token


@app.cell
def _(mo, start, token):
    import os, pathlib, subprocess
    mo.stop(not (start.value and token.value.strip()), mo.md("Waiting for the token and the Start training button."))
    kaggle_dir = pathlib.Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    (kaggle_dir / "access_token").write_text(token.value.strip())
    os.chmod(kaggle_dir / "access_token", 0o600)
    cmd = "python -m pip -q install -U kaggle && mkdir -p /kaggle/jobs && for f in job.py job.json molab_worker.py fetch_runs.py; do kaggle datasets download -d cocofioren/indabax-molab-jobs -f $f -p /kaggle/jobs -o -q || exit 1; done && python -u /kaggle/jobs/molab_worker.py"
    proc = subprocess.Popen(cmd, shell=True, env=dict(os.environ, KAGGLE_API_TOKEN=token.value.strip()), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    return


if __name__ == "__main__":
    app.run()
