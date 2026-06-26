$ErrorActionPreference = "Stop"

$repoUrl = "https://github.com/itsromath/easyrepet-pipeline.git"

$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    $gitCmdPath = "C:\Program Files\Git\cmd"
    if (Test-Path (Join-Path $gitCmdPath "git.exe")) {
        $env:Path = "$gitCmdPath;$env:Path"
    } else {
        Write-Host "Git is not available in PATH. Install Git for Windows, then reopen PowerShell and run this script again."
        exit 1
    }
}

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".git\config")) {
    git init
}

if (-not (git config user.name)) {
    git config user.name "itsromath"
}

if (-not (git config user.email)) {
    git config user.email "197004766+itsromath@users.noreply.github.com"
}

git add .

$hasCommit = (Test-Path ".git\refs\heads\main") -or (Test-Path ".git\refs\heads\master")

if ($hasCommit) {
    git commit -m "Update EasyRepet pipeline"
} else {
    git commit -m "Initial EasyRepet pipeline MVP"
}

git branch -M main

$hasOrigin = (git remote) -contains "origin"
if (-not $hasOrigin) {
    git remote add origin $repoUrl
    $remote = $repoUrl
} else {
    $remote = git remote get-url origin
}

if ($remote -ne $repoUrl) {
    git remote set-url origin $repoUrl
}

git push -u origin main
