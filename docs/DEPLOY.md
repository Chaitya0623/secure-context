# Deploying for the team

## Option 1: Streamlit Community Cloud (recommended, free)

The full Python app, with live detection, tokens, minimization, the exposure ledger and the evaluation results.

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Create app, then pick repository `Chaitya0623/secure-context` (or your fork), branch `rishika`, main file `app/streamlit_app.py`.
3. Deploy. The first load takes about 20 seconds while it generates the synthetic data.
4. Optional real model: in the app's Settings, then Secrets, add
   ```toml
   ANTHROPIC_API_KEY = "..."
   LLM_MODEL = "<model id>"
   ```
   and add `anthropic` to `requirements.txt`. Without a key the app uses the simulated model.

The repo must be public, or you must grant Streamlit access to it. Anyone with the link can use the app; the data is all synthetic.

## Option 2: Hugging Face Spaces

Create a Space with the Streamlit SDK, push this repo to it, and set `app_file: app/streamlit_app.py` in the Space README front matter. Secrets go under Settings, then Variables and secrets.

## Option 3: static walkthrough only

`prototype/index.html` is a single file with no backend. Host it with GitHub Pages (Settings, then Pages, deploy from branch, folder `/prototype`; needs admin on the repo) or open it locally.

## Pushing this branch

```bash
git remote add origin https://github.com/Chaitya0623/secure-context.git
git push origin main          # empty starter commit, so main stays the default branch
git push origin rishika
```

To also show it under your own profile, fork the repo on GitHub after the first push, then:

```bash
git remote add personal https://github.com/rishika1099/secure-context.git
git push personal rishika
```

Open a pull request from `rishika` into `main` so the commits count toward your contribution graph once merged.
