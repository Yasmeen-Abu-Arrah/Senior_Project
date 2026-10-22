# Create a Python 3.11 virtual environment and install all project requirements.
# Usage:
#   .\setup_venv.ps1          # auto: CUDA 12.8 wheels when nvidia-smi is found
#   .\setup_venv.ps1 -Cpu     # force CPU-only PyTorch
#   .\setup_venv.ps1 -Recreate
param(
    [switch]$Cpu,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$Venv = Join-Path $PSScriptRoot ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Habibi = Join-Path $PSScriptRoot "third_party\Habibi-TTS"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')"
    }
}

if ($Recreate -and (Test-Path -LiteralPath $Venv)) {
    Write-Host "Removing existing .venv..."
    Remove-Item -LiteralPath $Venv -Recurse -Force
}

if (Test-Path -LiteralPath $Py) {
    $ExistingVersion = (& $Py -c "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))").Trim()
    if ($ExistingVersion -ne "3.11") {
        Write-Host "Existing .venv uses Python $ExistingVersion; recreating with Python 3.11..."
        Remove-Item -LiteralPath $Venv -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath $Py)) {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        Write-Host "Creating Python 3.11 environment with uv..."
        Invoke-Checked -Exe $uv.Source -Arguments @("venv", "--python", "3.11", "--seed", $Venv)
    }
    else {
        $created = $false

        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($pyLauncher) {
            & py -3.11 -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Creating Python 3.11 environment with py launcher..."
                Invoke-Checked -Exe "py" -Arguments @("-3.11", "-m", "venv", $Venv)
                $created = $true
            }
        }

        if (-not $created) {
            foreach ($candidate in @("python3.11", "python")) {
                $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
                if (-not $cmd) { continue }
                & $cmd.Source -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "Creating Python 3.11 environment with $candidate..."
                    Invoke-Checked -Exe $cmd.Source -Arguments @("-m", "venv", $Venv)
                    $created = $true
                    break
                }
            }
        }

        if (-not $created) {
            throw "Python 3.11 was not found. Install Python 3.11 (or uv) and run this script again."
        }
    }
}

Write-Host "`nUsing interpreter: $Py"
Invoke-Checked -Exe $Py -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

$CpuOnly = [bool]$Cpu
if (-not $CpuOnly -and -not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $CpuOnly = $true
}

if ($CpuOnly) {
    Write-Host "`nInstalling CPU PyTorch 2.8.0..."
    Invoke-Checked -Exe $Py -Arguments @("-m", "pip", "install", "torch==2.8.0", "torchaudio==2.8.0")
}
else {
    Write-Host "`nInstalling CUDA 12.8 PyTorch 2.8.0..."
    Invoke-Checked -Exe $Py -Arguments @(
        "-m", "pip", "install",
        "--index-url", "https://download.pytorch.org/whl/cu128",
        "torch==2.8.0", "torchaudio==2.8.0"
    )
}

Write-Host "`nInstalling requirements..."
Invoke-Checked -Exe $Py -Arguments @("-m", "pip", "install", "-r", (Join-Path $PSScriptRoot "requirements.txt"))

Write-Host "`nInstalling local Habibi-TTS package..."
Invoke-Checked -Exe $Py -Arguments @("-m", "pip", "install", "--no-deps", "-e", $Habibi)

& $Py -m pip check
if ($LASTEXITCODE -ne 0) {
    Write-Warning "pip check reported dependency issues (see above)."
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg was not found on PATH. Install it before using Whisper or audio decoding."
}

Write-Host "`nSetup complete."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Download models/fonts with: python scripts\download_models.py all"
Write-Host "Run prototypes from prototypes\<name>\README.md"
if ($CpuOnly) { Write-Host "PyTorch: CPU build" } else { Write-Host "PyTorch: CUDA 12.8 build" }
