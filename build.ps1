$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Get-Command go -ErrorAction SilentlyContinue)) { throw '請先安裝 Go 1.24 或更新版本：https://go.dev/dl/' }
New-Item -ItemType Directory -Force tools,fonts,.build | Out-Null
function Compress-GzipFile($InputPath,$OutputPath) {
  $src=[IO.File]::OpenRead($InputPath)
  $dst=[IO.File]::Create($OutputPath)
  $gz=[IO.Compression.GzipStream]::new($dst,[IO.Compression.CompressionLevel]::Optimal)
  try { $src.CopyTo($gz) } finally { $gz.Dispose();$dst.Dispose();$src.Dispose() }
}
if (-not (Test-Path 'tools/ffmpeg.exe.gz')) {
  $archive=Join-Path $PSScriptRoot '.build/ffmpeg.zip'
  Invoke-WebRequest 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-essentials_build.zip' -OutFile $archive
  if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLower() -ne '60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba') {throw 'FFmpeg SHA256 不符，已停止。'}
  Expand-Archive $archive -DestinationPath '.build/ffmpeg' -Force
  $engine=Get-ChildItem '.build/ffmpeg' -Filter ffmpeg.exe -Recurse | Select-Object -First 1
  if (-not $engine) {throw '找不到 ffmpeg.exe'}
  Compress-GzipFile $engine.FullName (Join-Path $PSScriptRoot 'tools/ffmpeg.exe.gz')
}
if (-not (Test-Path 'fonts/NotoSansTC.ttf.gz')) {
  $font=Join-Path $PSScriptRoot '.build/NotoSansTC.ttf'
  Invoke-WebRequest 'https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/Variable/TTF/Subset/NotoSansTC-VF.ttf' -OutFile $font
  if ((Get-FileHash $font -Algorithm SHA256).Hash.ToLower() -ne 'ac091cc8cd19e848202afc8fe6d3809b4526c8fdbdb4be82da20c4f785949591') {throw '字型 SHA256 不符，已停止；請確認上游檔案是否更新。'}
  Compress-GzipFile $font (Join-Path $PSScriptRoot 'fonts/NotoSansTC.ttf.gz')
}
$env:CGO_ENABLED='0';$env:GOOS='windows';$env:GOARCH='amd64'
go build -buildvcs=false -trimpath -ldflags '-s -w -H windowsgui' -o JunbaTalkVideo.exe .
if ($LASTEXITCODE -ne 0) {throw '編譯失敗'}
Write-Host '完成：JunbaTalkVideo.exe（單一 Windows 64 位元程式）'
