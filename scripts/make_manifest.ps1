$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$out = Join-Path $root "ARTIFACT_FILES.txt"

$files = Get-ChildItem -LiteralPath $root -Recurse -File |
  Where-Object { $_.FullName -notmatch "\\.git\\" } |
  ForEach-Object {
    $_.FullName.Substring($root.Length + 1).Replace("\", "/")
  } |
  Sort-Object

[System.IO.File]::WriteAllLines($out, [string[]]$files, [System.Text.UTF8Encoding]::new($false))

Write-Host "Wrote $out"
