param(
    [string]$BaseName = 'Calmets_Kilian',
    [string]$OutputDirectory = 'Rendu_final_Calmets_Pezet'
)

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputDir = Join-Path $project $OutputDirectory
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$output = Join-Path $outputDir "$BaseName.zip"
$items = @(
    '.gitignore', 'README.md', 'requirements.txt',
    'train.py', 'evaluate.py', 'metrics.py', 'report.py',
    'figures.py', 'dashboard.py', 'make_submission.ps1',
    'ouvrir_interface.ps1', 'ouvrir_interface.cmd',
    'fermer_interface.ps1', 'fermer_interface.cmd',
    'agent/__init__.py', 'agent/network.py', 'agent/replay.py',
    'figures',
    'runs/final_report.json',
    'runs/double_nstep_1m_g4_seed1/config.json', 'runs/double_nstep_1m_g4_seed1/scalars.csv',
    'runs/double_nstep_1m_g4_seed2/config.json', 'runs/double_nstep_1m_g4_seed2/scalars.csv',
    'runs/final_1m_g4_seed1/episodes.jsonl', 'runs/final_1m_g4_seed1/summary.json',
    'runs/final_1m_g4_seed2/episodes.jsonl', 'runs/final_1m_g4_seed2/summary.json',
    'runs/random_seed7001/episodes.jsonl', 'runs/random_seed7001/summary.json',
    'runs/behavior_1m_seed1/trace.jsonl'
)

Add-Type -AssemblyName System.IO.Compression.FileSystem
if (Test-Path -LiteralPath $output) { Remove-Item -LiteralPath $output }
$archive = [System.IO.Compression.ZipFile]::Open($output, 'Create')
try {
    foreach ($item in $items) {
        $path = Join-Path $project $item
        $files = if (Test-Path -LiteralPath $path -PathType Container) {
            Get-ChildItem -LiteralPath $path -Recurse -File
        } else {
            Get-Item -LiteralPath $path
        }
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($project.Length + 1).Replace('\', '/')
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $file.FullName, $relative,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
} finally {
    $archive.Dispose()
}
Write-Output $output
