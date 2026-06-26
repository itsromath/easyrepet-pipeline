param(
    [string]$Action = "menu"
)

$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent $PSScriptRoot
$LogsDir = Join-Path $Root "logs"
$PidFile = Join-Path $LogsDir "flask_server.pid"
$FlaskOut = Join-Path $LogsDir "flask_server_menu.log"
$FlaskErr = Join-Path $LogsDir "flask_server_menu.err.log"
$VenvPythonPath = Join-Path $Root ".venv\Scripts\python.exe"
$RuntimePythonPath = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PythonPath = if (Test-Path -LiteralPath $RuntimePythonPath) { $RuntimePythonPath } else { $VenvPythonPath }
$PythonSitePackages = Join-Path $Root ".venv\Lib\site-packages"
$LmsPath = Join-Path $env:USERPROFILE ".lmstudio\bin\lms.exe"
$DockerDesktopPath = if ($env:EASYREPET_DOCKER_DESKTOP) {
    $env:EASYREPET_DOCKER_DESKTOP
} else {
    @(
        (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"),
        (Join-Path $env:LOCALAPPDATA "Docker\Docker Desktop.exe")
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
$LmStudioAppPath = if ($env:EASYREPET_LM_STUDIO_APP) {
    $env:EASYREPET_LM_STUDIO_APP
} else {
    @(
        (Join-Path $env:LOCALAPPDATA "Programs\LM Studio\LM Studio.exe"),
        (Join-Path $env:LOCALAPPDATA "LM Studio\LM Studio.exe"),
        (Join-Path $env:ProgramFiles "LM Studio\LM Studio.exe")
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
$SpeachesContainer = if ($env:EASYREPET_SPEACHES_CONTAINER) { $env:EASYREPET_SPEACHES_CONTAINER } else { "speaches" }
$FlaskUrl = "http://127.0.0.1:5050"
$SpeachesModelsUrl = "http://127.0.0.1:8000/v1/models"
$LmModelsUrl = "http://127.0.0.1:1234/v1/models"
$ModelPresetsPath = Join-Path $Root "config\model_presets.json"
$FallbackLmModels = @("qwen/qwen3-4b-2507", "qwen/qwen3.5-9b")
$DockerWaitSeconds = 120
$LmStudioWaitSeconds = 60

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

function Write-Section([string]$Text) {
    Write-Host ""
    Write-Host "== $Text =="
}

function Test-Http([string]$Url) {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Get-JsonOrNull([string]$Url) {
    try {
        return Invoke-RestMethod -Uri $Url -TimeoutSec 3
    } catch {
        return $null
    }
}

function Get-LmPresetModels {
    if (Test-Path -LiteralPath $ModelPresetsPath) {
        try {
            $presets = Get-Content -Raw -Encoding UTF8 -LiteralPath $ModelPresetsPath | ConvertFrom-Json
            $models = @()
            foreach ($property in $presets.PSObject.Properties) {
                if ($property.Value.model) {
                    $models += [string]$property.Value.model
                }
            }
            if ($models.Count -gt 0) {
                return @($models | Select-Object -Unique)
            }
        } catch {
            Write-Host "Could not read model presets: $($_.Exception.Message)"
        }
    }

    return $FallbackLmModels
}

function Get-FlaskStatusJson {
    return Get-JsonOrNull "$FlaskUrl/api/status"
}

function Test-FlaskCurrent {
    $json = Get-FlaskStatusJson
    return ($json -and ($json.PSObject.Properties.Name -contains "stage"))
}

function Wait-ForCondition([string]$Label, [int]$TimeoutSeconds, [scriptblock]$Condition) {
    Write-Host "Waiting for $Label..."
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (& $Condition) {
            Write-Host "$Label is ready."
            return $true
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }

    Write-Host ""
    Write-Host "$Label is not ready after $TimeoutSeconds seconds."
    return $false
}

function Start-ExternalApp([string]$Path, [string]$Label) {
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) {
        Write-Host "$Label path not found."
        return $false
    }

    try {
        Write-Host "Opening $Label..."
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $Path
        $startInfo.UseShellExecute = $true
        $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Minimized
        [System.Diagnostics.Process]::Start($startInfo) | Out-Null
        return $true
    } catch {
        Write-Host "Could not open ${Label}: $($_.Exception.Message)"
        return $false
    }
}

function Test-AnyProcess([string[]]$Names) {
    foreach ($name in $Names) {
        if (Get-Process -Name $name -ErrorAction SilentlyContinue) {
            return $true
        }
    }
    return $false
}

function Test-DockerReady {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        return $false
    }

    & docker info *> $null
    return ($LASTEXITCODE -eq 0)
}

function Start-DockerDesktop {
    if (Test-DockerReady) {
        Write-Host "Docker Engine is already ready."
        return $true
    }

    if (-not (Test-AnyProcess @("Docker Desktop", "com.docker.backend"))) {
        Start-ExternalApp $DockerDesktopPath "Docker Desktop" | Out-Null
    } else {
        Write-Host "Docker Desktop is already opening."
    }

    return Wait-ForCondition "Docker Engine" $DockerWaitSeconds { Test-DockerReady }
}

function Start-LmStudioApp {
    if (Test-AnyProcess @("LM Studio", "LM Studio Helper")) {
        Write-Host "LM Studio app is already open."
        return $true
    }

    if (-not (Start-ExternalApp $LmStudioAppPath "LM Studio")) {
        return $false
    }

    return Wait-ForCondition "LM Studio app" 30 { Test-AnyProcess @("LM Studio", "LM Studio Helper") }
}

function Stop-PidTree([int]$ProcessId) {
    & taskkill.exe /PID $ProcessId /T /F 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -eq 0) {
        return $true
    }

    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
        return $true
    } catch {
        Write-Host "Could not stop PID $ProcessId. Run this menu as administrator if Windows denies access."
        return $false
    }
}

function Get-PortListenerPids([int]$Port) {
    $pids = @()

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        if ($conn.OwningProcess) {
            $pids += [int]$conn.OwningProcess
        }
    }

    $lines = & netstat -ano -p tcp 2>$null | Select-String "LISTENING" | Select-String ":$Port"
    foreach ($line in $lines) {
        $parts = ($line.Line.Trim() -split "\s+")
        if ($parts.Count -ge 5 -and $parts[1] -match ":$Port$") {
            $pids += [int]$parts[4]
        }
    }

    return @($pids | Sort-Object -Unique)
}

function Stop-FlaskPortListener {
    $pids = Get-PortListenerPids 5050
    foreach ($pidValue in $pids) {
        if (Stop-PidTree $pidValue) {
            Write-Host "Stopped process on port 5050. PID: $pidValue"
        }
    }

    if ($pids.Count -gt 0) {
        Start-Sleep -Seconds 1
        $remaining = Get-PortListenerPids 5050
        if ($remaining.Count -gt 0) {
            Write-Host "Port 5050 is still busy: $($remaining -join ', ')"
            Write-Host "Run EasyRepet menu as administrator if Windows denies access."
            return $false
        }
        return $true
    }

    return $false
}

function Get-FlaskProcess {
    if (-not (Test-Path -LiteralPath $PidFile)) {
        return $null
    }

    $pidText = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $pidText) {
        return $null
    }

    try {
        return Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
    } catch {
        return $null
    }
}

function Start-Speaches {
    Write-Section "Speaches Docker"
    if (Test-Http $SpeachesModelsUrl) {
        Write-Host "Speaches API is already ready: $SpeachesModelsUrl"
        return
    }

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Host "Docker CLI not found."
        return
    }

    if (-not (Start-DockerDesktop)) {
        Write-Host "Docker Desktop is not ready, so Speaches container may not start."
        return
    }

    & docker start $SpeachesContainer
    if (Wait-ForCondition "Speaches API" 45 { Test-Http $SpeachesModelsUrl }) {
        Write-Host "Speaches API is ready: $SpeachesModelsUrl"
    } else {
        Write-Host "Speaches started, but API is not ready yet: $SpeachesModelsUrl"
    }
}

function Start-LmStudio {
    Write-Section "LM Studio"
    if (-not (Test-Path -LiteralPath $LmsPath)) {
        Write-Host "lms.exe not found: $LmsPath"
        return
    }

    if (Test-Http $LmModelsUrl) {
        Write-Host "LM Studio API is already ready: $LmModelsUrl"
    } else {
        if (-not (Start-LmStudioApp)) {
            Write-Host "LM Studio app was not opened automatically."
            Write-Host "Open LM Studio manually, then run Start again."
            return
        }

        & $LmsPath server start
        Wait-ForCondition "LM Studio server" $LmStudioWaitSeconds { Test-Http $LmModelsUrl } | Out-Null
    }

    $lmJson = Get-JsonOrNull $LmModelsUrl
    if ($lmJson) {
        Write-Host "LM Studio API is ready: $LmModelsUrl"
        $ids = @()
        if ($lmJson.data) {
            $ids = @($lmJson.data | ForEach-Object { $_.id })
        }
        $modelsToCheck = Get-LmPresetModels
        foreach ($model in $modelsToCheck) {
            if ($ids -contains $model) {
                Write-Host "Model available to API: $model"
            } else {
                Write-Host "Model is not currently listed by LM Studio API: $model"
                Write-Host "It may still be JIT-loadable if LM Studio knows this model."
                Write-Host "Try sending a test request or open LM Studio and check the model identifier."
            }
        }
    } else {
        Write-Host "LM Studio server was requested, but API is not ready yet: $LmModelsUrl"
    }
}

function Start-Flask {
    Write-Section "Flask"
    $existing = Get-FlaskProcess
    if ($existing) {
        Write-Host "Flask already started by this menu. PID: $($existing.Id)"
        return
    }

    $flaskStatus = Get-FlaskStatusJson
    if ($flaskStatus) {
        if ($flaskStatus.PSObject.Properties.Name -contains "stage") {
            Write-Host "Flask already responds with current API at $FlaskUrl."
            $listenerPids = Get-PortListenerPids 5050
            if ($listenerPids.Count -gt 0) {
                Set-Content -LiteralPath $PidFile -Value $listenerPids[0]
            }
            return
        }

        Write-Host "Port 5050 responds, but it looks like an old EasyRepet Flask API."
        if (-not (Stop-FlaskPortListener)) {
            Write-Host "Could not find the listener PID automatically."
            Write-Host "Close the old Flask window/process or reboot Windows, then run Start again."
            return
        }
    }

    if (-not (Test-Path -LiteralPath $PythonPath)) {
        Write-Host "Python venv not found: $PythonPath"
        return
    }

    $localPath = "$Root\.venv\Scripts;$env:SystemRoot\System32;$env:SystemRoot"
    Set-Content -LiteralPath $FlaskOut -Value ""
    Set-Content -LiteralPath $FlaskErr -Value ""

    $command = "set ""PATH=$localPath"" && set ""PYTHONPATH=$PythonSitePackages"" && set ""PYTHONUTF8=1"" && set ""PYTHONIOENCODING=utf-8"" && set ""EASYREPET_FLASK_DEBUG=0"" && set ""EASYREPET_PORT=5050"" && ""$PythonPath"" app.py >> ""$FlaskOut"" 2>> ""$FlaskErr"""

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = "cmd.exe"
    $startInfo.Arguments = "/d /c $command"
    $startInfo.WorkingDirectory = $Root
    $startInfo.UseShellExecute = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $process = [System.Diagnostics.Process]::Start($startInfo)

    if (-not $process) {
        Write-Host "Could not start Flask process."
        return
    }

    Set-Content -LiteralPath $PidFile -Value $process.Id
    Write-Host "Flask PID: $($process.Id)"

    $ready = $false
    for ($i = 0; $i -lt 15; $i++) {
        if (Test-FlaskCurrent) {
            $ready = $true
            break
        }
        Start-Sleep -Seconds 1
    }

    if ($ready) {
        Write-Host "Flask is ready: $FlaskUrl"
    } else {
        Write-Host "Flask started, but is not responding yet."
        if (Test-Path -LiteralPath $FlaskErr) {
            Write-Host ""
            Write-Host "Last Flask error log lines:"
            Get-Content -LiteralPath $FlaskErr -Tail 20 -ErrorAction SilentlyContinue
        }
    }
}

function Start-EasyRepet {
    Start-Speaches
    Start-LmStudio
    Start-Flask
    Open-EasyRepet
}

function Stop-Flask {
    Write-Section "Flask"
    $process = Get-FlaskProcess
    if ($process) {
        if (Stop-PidTree $process.Id) {
            Write-Host "Stopped Flask PID: $($process.Id)"
        }
        Stop-FlaskPortListener | Out-Null
    } else {
        Write-Host "No Flask PID from this menu is running."
        Stop-FlaskPortListener | Out-Null
    }
}

function Stop-Speaches {
    Write-Section "Speaches Docker"
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Host "Docker CLI not found."
        return
    }

    & docker stop $SpeachesContainer
}

function Stop-LmStudioServer {
    Write-Section "LM Studio server"
    if (-not (Test-Path -LiteralPath $LmsPath)) {
        Write-Host "lms.exe not found: $LmsPath"
        return
    }

    & $LmsPath server stop
}

function Stop-EasyRepet {
    Stop-Flask
    Stop-Speaches
    Stop-LmStudioServer
}

function Show-Status {
    Write-Section "EasyRepet status"

    $flaskProcess = Get-FlaskProcess
    if ($flaskProcess) {
        Write-Host "Flask PID: $($flaskProcess.Id)"
    } else {
        Write-Host "Flask PID: not started by this menu"
    }

    $flaskStatus = Get-FlaskStatusJson
    if ($flaskStatus) {
        if ($flaskStatus.PSObject.Properties.Name -contains "stage") {
            Write-Host "Flask API: OK, current version"
        } else {
            Write-Host "Flask API: OLD version on 5050"
        }
    } else {
        Write-Host "Flask API: not responding"
    }

    if (Test-Http $SpeachesModelsUrl) {
        Write-Host "Speaches API: OK"
    } else {
        Write-Host "Speaches API: not responding"
    }

    $lmJson = Get-JsonOrNull $LmModelsUrl
    if ($lmJson) {
        Write-Host "LM Studio API: OK"
        $ids = @()
        if ($lmJson.data) {
            $ids = @($lmJson.data | ForEach-Object { $_.id })
        }
        $modelsToCheck = Get-LmPresetModels
        foreach ($model in $modelsToCheck) {
            if ($ids -contains $model) {
                Write-Host "Model available to API: $model"
            } else {
                Write-Host "Model is not currently listed by LM Studio API: $model"
                Write-Host "It may still be JIT-loadable if LM Studio knows this model."
                Write-Host "Try sending a test request or open LM Studio and check the model identifier."
            }
        }
    } else {
        Write-Host "LM Studio API: not responding"
    }
}

function Open-EasyRepet {
    Write-Section "Browser"
    try {
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $FlaskUrl
        $startInfo.UseShellExecute = $true
        [System.Diagnostics.Process]::Start($startInfo) | Out-Null
    } catch {
        Write-Host "Open manually: $FlaskUrl"
    }
}

function Show-Menu {
    while ($true) {
        Write-Host ""
        Write-Host "EasyRepet"
        Write-Host "1. Start"
        Write-Host "2. Stop"
        Write-Host "3. Status"
        Write-Host "4. Open Browser"
        Write-Host "5. Exit"
        $choice = Read-Host "Choose"

        switch ($choice) {
            "1" { Start-EasyRepet }
            "2" { Stop-EasyRepet }
            "3" { Show-Status }
            "4" { Open-EasyRepet }
            "5" { return }
            default { Write-Host "Unknown choice." }
        }
    }
}

switch ($Action.ToLowerInvariant()) {
    "start" { Start-EasyRepet }
    "stop" { Stop-EasyRepet }
    "status" { Show-Status }
    "open" { Open-EasyRepet }
    default { Show-Menu }
}
