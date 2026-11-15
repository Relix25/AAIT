$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidPath = Join-Path $projectRoot '.interface\server.pid'
$port = 8517
if (-not (Test-Path -LiteralPath $pidPath)) {
    Write-Output "Aucune interface lancée par ce projet."
    exit 0
}

$serverId = [int](Get-Content -LiteralPath $pidPath -Raw)
$listener = netstat -ano | Where-Object { $_ -match "^\s*TCP\s+127\.0\.0\.1:$port\s+.*LISTENING\s+(\d+)\s*$" } | Select-Object -First 1
$listenerId = if ($listener -match 'LISTENING\s+(\d+)\s*$') { [int]$Matches[1] } else { 0 }
$processInfo = Get-Process -Id $serverId -ErrorAction SilentlyContinue
if ($listenerId -eq $serverId -and $processInfo -and $processInfo.ProcessName -eq 'python') {
    Stop-Process -Id $serverId
    Write-Output "Interface arrêtée."
} else {
    Write-Output "Aucun processus Streamlit correspondant à arrêter."
}
Remove-Item -LiteralPath $pidPath
