# GitHub Sync Plan

## Private Development Repository

Use a private GitHub repository for development cleanup and collaborator review.

Suggested name:

```text
tecs-clahe-artifact-private
```

## Anonymous Review Repository

For double-blind review, do not fork from the private repository and do not keep
development history. Export a clean working tree, then initialize a new Git
history using anonymous author metadata.

Suggested public or private review name:

```text
tecs-clahe-artifact
```

## Before Pushing

Run:

```powershell
pwsh scripts/scan_artifact.ps1
pwsh scripts/make_manifest.ps1
git status
```

## GitHub CLI Commands

If GitHub CLI is installed and authenticated:

```powershell
git init
git add .
git commit -m "Initial private review artifact"
gh repo create tecs-clahe-artifact-private --private --source=. --remote=origin --push
```

GitHub CLI is not currently available in this Codex shell, so the remote step
must be done later through GitHub Desktop, the GitHub web UI, or an installed
`gh` command.

