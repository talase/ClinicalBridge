# GitHub Release Checklist

## Files Included

- [x] `Agent1 triage.py`
- [x] `Agent2 ehr.py`
- [x] `Agent3 anamnesis.py`
- [x] `Agent4 synthesis.py`
- [x] `prompt_test_cases.py`
- [x] `run_prompt_tests.py`
- [x] `prompt_iteration_log.md`
- [x] `Prompt_Engineering_Portfolio.pdf`
- [x] `README.md`
- [x] `requirements.txt`
- [x] `.gitignore`
- [x] `RELEASE_CHECKLIST.md`

## Files Excluded

- [x] `.venv/`
- [x] `__pycache__/`
- [x] `*.pyc`
- [x] `.DS_Store`
- [x] `.env` and `.env.*`
- [x] `.pytest_cache/`
- [x] `.mypy_cache/`
- [x] `.idea/`
- [x] `.vscode/`
- [x] Local configuration, credential, token, certificate, and secret files

## Security Review

- [x] Repository scanned for OpenAI key patterns.
- [x] Repository scanned for password, secret, token, and credential references.
- [x] No hard-coded API secrets found.
- [x] OpenAI credentials are read with `os.getenv("OPENAI_API_KEY")`.
- [x] Secret-bearing environment files are excluded by `.gitignore`.

## Verification

- [x] All Python files compile successfully.
- [x] Existing local Pydantic smoke tests pass.
- [x] Cross-agent test runner skips cleanly when no API key is configured.
- [ ] Live LLM regression tests require a valid API key and available API quota.

## Publication Commands

Replace the GitHub username and repository name in the remote URL, then run:

```bash
git init
git branch -M main
git add .
git status
git commit -m "Initial release: clinical prompt engineering portfolio"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

Before committing, confirm that `git status` does not list `.venv`, `__pycache__`,
`.env`, IDE settings, or any credential file.

