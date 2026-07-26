[CmdletBinding()]
param(
    [string]$Destination = "fork/personaplex-agent-moshi"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$DestinationPath = Join-Path $Root $Destination
$PatchRoot = Join-Path $Root "personaplex_moshi_patch\moshi"
$PinnedCommit = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"

if (-not (Test-Path -LiteralPath $DestinationPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $DestinationPath) | Out-Null
    git clone https://github.com/NVIDIA/personaplex.git $DestinationPath
}

git -C $DestinationPath fetch origin main
git -C $DestinationPath checkout --detach $PinnedCommit

$MoshiRoot = Join-Path $DestinationPath "moshi\moshi"
Copy-Item (Join-Path $PatchRoot "models\*.py") (Join-Path $MoshiRoot "models") -Force
Copy-Item (Join-Path $PatchRoot "modules\transformer.py") (Join-Path $MoshiRoot "modules\transformer.py") -Force

Write-Host "PersonaPlex patch installed at $DestinationPath"
Write-Host "Pinned upstream commit: $PinnedCommit"
