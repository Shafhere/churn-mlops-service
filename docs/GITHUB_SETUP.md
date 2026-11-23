# Publish ChurnGuard to GitHub (source only)

The GitHub repository `Shafhere/churn-mlops-service` exists but the uploaded snapshot did not establish that this cleaned release has been pushed. The ZIP includes **no `.git` directory**. Publish only after checking the staged file list.

1. Extract the release ZIP and open Command Prompt/PowerShell inside `ChurnGuard_Final_Ready`.
2. While preparing the machine task, consider changing the repository visibility to **private** in GitHub Settings → General → Danger Zone. The repository was visible as public when checked; never assume it is private.
3. Initialize the source repository if needed:

```cmd
git init -b main
git remote add origin https://github.com/Shafhere/churn-mlops-service.git
git add -A
git status
```

4. **Before committing**, inspect `git status` and ensure `.env`, `data/raw`, `artifacts`, MLflow databases, caches and monitoring `.sqlite` files are **not staged**. They are ignored by `.gitignore`; the AWS release ZIP deliberately includes the required approved sample model artifact separately.
5. Push source:

```cmd
git commit -m "feat: deliver reviewed churn prediction MLOps service"
git push -u origin main
```

If the repo already contains an unrelated initial commit, **do not force push**. First compare existing history and reconcile commits. Don't insert passwords/tokens into Git remote URLs.

6. Visit <https://github.com/Shafhere/churn-mlops-service/actions> and confirm that Ruff and Pytest pass. The GitHub workflow tests *source code* with synthetic data and does not need raw dataset or trained model artifacts. GitHub hosting of files is **not** deployment of the UI.

## Git workflow explanation for interview

Every change is committed with a descriptive message; CI tests code quality and behavior. Model artifacts are separately versioned/tracked with MLflow and the release manifest. The serving deployment is a deliberate release step, not an automatic push of every branch to AWS.
