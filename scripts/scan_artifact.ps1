$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$self = (Resolve-Path -LiteralPath $MyInvocation.MyCommand.Path).Path
$textExtensions = @(
  ".cff", ".csv", ".json", ".md", ".ps1", ".py", ".sv", ".svh",
  ".tcl", ".toml", ".txt", ".v", ".vh", ".yml", ".yaml"
)
$patterns = @(
  "PERSONAL_GITHUB_USERNAME",
  "C:\\Users",
  "F:\\EngineeringWarehouse",
  "D:\\Program Files",
  "apiKey\\s*[:=]",
  "password\\s*[:=]",
  "secret\\s*[:=]",
  "token\\s*[:=]",
  "@gmail",
  "@qq",
  "@outlook"
)

$hits = @()
foreach ($pattern in $patterns) {
  $files = Get-ChildItem -LiteralPath $root -Recurse -File |
    Where-Object {
      $_.FullName -notmatch "\\.git\\" -and
      $_.FullName -ne $self -and
      $textExtensions -contains $_.Extension.ToLowerInvariant()
    }
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
