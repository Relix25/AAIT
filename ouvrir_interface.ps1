$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$logDirectory = Join-Path $projectRoot '.interface'
$port = 8517
$url = "http://127.0.0.1:$port"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python local absent. Installez les dépendances du README avant d'ouvrir l'interface."
}

function Test-Interface {
    try {
        $response = Invoke-WebRequest -Uri "$url/_stcore/health" -UseBasicParsing -TimeoutSec 1
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (-not (Test-Interface)) {
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $env:MPLCONFIGDIR = Join-Path $logDirectory 'matplotlib'
    New-Item -ItemType Directory -Path $env:MPLCONFIGDIR -Force | Out-Null
    $server = Start-Process -FilePath $pythonPath -ArgumentList @(
        '-m', 'streamlit', 'run', 'dashboard.py',
        '--server.headless=true', '--server.address=127.0.0.1',
        "--server.port=$port", '--browser.gatherUsageStats=false'
    ) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput (Join-Path $logDirectory 'server.out.log') `
      -RedirectStandardError (Join-Path $logDirectory 'server.err.log')
    for ($attempt = 0; $attempt -lt 40 -and -not (Test-Interface); $attempt++) {
        Start-Sleep -Milliseconds 500
    }
}

if (-not (Test-Interface)) {
    throw "L'interface n'a pas démarré. Voir .interface\server.err.log dans le projet."
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$listener = netstat -ano | Where-Object { $_ -match "^\s*TCP\s+127\.0\.0\.1:$port\s+.*LISTENING\s+(\d+)\s*$" } | Select-Object -First 1
if ($listener -match 'LISTENING\s+(\d+)\s*$') {
    Set-Content -LiteralPath (Join-Path $logDirectory 'server.pid') -Value $Matches[1]
}

try {
    Start-Process $url
} catch {
    Write-Warning "Le navigateur ne s'est pas ouvert automatiquement. Ouvrez $url dans votre navigateur."
}
Write-Output "Interface ouverte : $url"
