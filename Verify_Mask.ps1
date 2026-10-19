param(
  [string]$InputPath,
  [ValidateSet('auto','real','cartoon')][string]$Style = 'auto',
  [ValidateSet(3,4,5)][int]$Feather = 4
)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
try {
  $baseRuntime=Join-Path $env:APPDATA 'JunbaTalkVideo\local-v0.2'
  $pythonPath=Join-Path $baseRuntime 'python\python.exe'
  if (-not (Test-Path $pythonPath)) { $pythonPath=Join-Path $PSScriptRoot 'runtime\base\python\python.exe' }
  if (-not (Test-Path $pythonPath)) { throw '找不到既有 Junba Python 子模組。請先在原程式使用一鍵安裝；本工具不要求安裝系統 Python。' }
  if (-not $InputPath) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog=New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title='選擇測試照片，或既有專案的 local-config.json'
    $dialog.Filter='照片或專案設定|*.jpg;*.jpeg;*.png;*.webp;*.json'
    $projects=Join-Path $env:APPDATA 'JunbaTalkVideo\output'
    if (Test-Path $projects) { $dialog.InitialDirectory=$projects }
    if ($dialog.ShowDialog() -ne 'OK') { return }
    $InputPath=$dialog.FileName
  }
  $outPath=Join-Path $PSScriptRoot ('verification\'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
  $pythonArgs=@('--style',$Style,'--feather',"$Feather",'--out',$outPath)
  if ((Get-Item $InputPath).PSIsContainer) { $pythonArgs+=@('--project',$InputPath) }
  elseif ([IO.Path]::GetExtension($InputPath) -eq '.json') { $pythonArgs+=@('--config',$InputPath) }
  else { $pythonArgs+=@('--image',$InputPath,'--models',(Join-Path $baseRuntime 'models')) }
  & $pythonPath -I (Join-Path $PSScriptRoot 'verify_mask.py') @pythonArgs
  if ($LASTEXITCODE -ne 0) { throw '遮罩驗證未完成，請查看 verification 資料夾內日誌；未編譯或生成影片。' }
  Start-Process (Join-Path $outPath 'comparison.png')
  Write-Host '已輸出比對圖及 verification.json。收到您的 OK 前不會編譯 EXE。'
} catch {
  Write-Host $_.Exception.Message -ForegroundColor Red
  exit 1
}
