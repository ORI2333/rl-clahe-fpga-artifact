$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$patterns = @(
  "ORI2333",
  "C:\\Users",
  "F:\\EngineeringWarehouse",
  "D:\\Program Files",
  "Obsidian",
  "apiKey",
  "password",
  "secret",
  "token",
  "@gmail",
  "@qq",
  "@outlook"
)

$hits = @()
foreach ($pattern in $patterns) {
  $files = Get-ChildItem -LiteralPath $root -Recurse -File |
    Where-Object { $_.FullName -notmatch "\\.git\\" }
  $result = $files | Select-String -Pattern $pattern -ErrorAction SilentlyContinue
  if ($result) {
    $hits += $result
  }
}

if ($hits.Count -gt 0) {
  Write-Host "Potential privacy/release issues found:" -ForegroundColor Yellow
  $hits | ForEach-Object {
    "{0}:{1}: {2}" -f $_.Path, $_.LineNumber, $_.Line.Trim()
  }
  exit 1
}

Write-Host "No configured privacy patterns found."
