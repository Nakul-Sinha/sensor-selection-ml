<#
ship.ps1 -- one change = one branch = one PR = squash-merged to main.

GitHub's native "auto-merge" requires a paid plan for private repos, so this reproduces the same
outcome: branch, commit, push, open a PR, then squash-merge and delete the branch.

Usage:
    pwsh tools/ship.ps1 -Title "short title" -Body "what changed and why"
    pwsh tools/ship.ps1 -Title "..." -Body "..." -Paths "Challenge1/solution.py","Challenge1/FINDINGS.md"

If -Paths is omitted, every pending change is included.
#>
param(
    [Parameter(Mandatory = $true)][string]$Title,
    [string]$Body = "",
    [string[]]$Paths,
    [string]$Repo = "Nakul-Sinha/eris-challenges"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# a branch name derived from the title
$slug = ($Title.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
if ($slug.Length -gt 48) { $slug = $slug.Substring(0, 48).Trim('-') }
$stamp = Get-Date -Format "MMdd-HHmmss"
$branch = "change/$stamp-$slug"

git checkout -q main
git pull -q --ff-only origin main 2>$null
git checkout -q -b $branch

if ($Paths) { git add -- $Paths } else { git add -A }

$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Output "nothing to commit; aborting"
    git checkout -q main
    git branch -q -D $branch
    exit 0
}

# refuse to ship secrets, whatever .gitignore says
$bad = $staged | Where-Object { $_ -match "credential|cookie|secret|token|\.pem$|\.key$" }
if ($bad) {
    Write-Output "REFUSING TO SHIP - secret-looking files staged:"
    $bad | ForEach-Object { Write-Output "  $_" }
    git reset -q
    git checkout -q main
    git branch -q -D $branch
    exit 1
}

$msg = $Title
if ($Body) { $msg = "$Title`n`n$Body" }
$msg = "$msg`n`nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git commit -q -m $msg
git push -q -u origin $branch

$prBody = if ($Body) { $Body } else { $Title }
$prBody = "$prBody`n`n🤖 Generated with [Claude Code](https://claude.com/claude-code)"
gh pr create --repo $Repo --base main --head $branch --title $Title --body $prBody | Out-Null
gh pr merge --repo $Repo $branch --squash --delete-branch | Out-Null

git checkout -q main
git pull -q --ff-only origin main

Write-Output "shipped: $Title"
Write-Output ("files: " + ($staged -join ", "))
git log --oneline -1
