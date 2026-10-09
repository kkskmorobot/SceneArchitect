$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$targetDirectory = Join-Path $projectRoot "models\controlnet-depth-sdxl-1.0-small"
$partialPath = Join-Path $targetDirectory "diffusion_pytorch_model.fp16.safetensors.part"
$finalPath = Join-Path $targetDirectory "diffusion_pytorch_model.fp16.safetensors"
$configPath = Join-Path $targetDirectory "config.json"
$weightUrl = "https://hf-mirror.com/diffusers/controlnet-depth-sdxl-1.0-small/resolve/main/diffusion_pytorch_model.fp16.safetensors"
$configUrl = "https://hf-mirror.com/diffusers/controlnet-depth-sdxl-1.0-small/resolve/main/config.json"
$expectedSize = 320237179
$expectedHash = "6f23c58fd632f52238a7b35ebdc02f9f596fd13dbaa121f9b37b9f4689c2b1e9"

New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null

$weightReady = $false
if (Test-Path -LiteralPath $finalPath) {
    $currentHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $finalPath).Hash.ToLowerInvariant()
    $weightReady = $currentHash -eq $expectedHash
    if (-not $weightReady) {
        throw "正式权重文件存在，但 SHA-256 不匹配：$finalPath"
    }
}

if (-not $weightReady) {
    Write-Host "开始或继续下载 Depth ControlNet。网络中断后重新运行本脚本即可。"
    & curl.exe -L --fail --connect-timeout 30 --speed-time 120 --speed-limit 1024 `
        -C - -o $partialPath $weightUrl
    if ($LASTEXITCODE -ne 0) {
        Write-Host "下载连接已中断，断点文件已保留：$partialPath"
        Write-Host "请重新运行同一脚本继续，不要删除 .part 文件。"
        exit $LASTEXITCODE
    }

    $downloadedSize = (Get-Item -LiteralPath $partialPath).Length
    if ($downloadedSize -ne $expectedSize) {
        throw "文件大小异常：$downloadedSize，预期：$expectedSize"
    }
    $downloadedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $partialPath).Hash.ToLowerInvariant()
    if ($downloadedHash -ne $expectedHash) {
        throw "SHA-256 不匹配。保留 .part 文件供排查，不会替换正式权重。"
    }
    Move-Item -LiteralPath $partialPath -Destination $finalPath -Force
}

& curl.exe -L --fail --connect-timeout 30 -o $configPath $configUrl
if ($LASTEXITCODE -ne 0) {
    throw "权重已就绪，但 config.json 下载失败；重新运行本脚本即可。"
}

Write-Host "Depth ControlNet 下载及校验完成："
Get-Item -LiteralPath $finalPath, $configPath | Select-Object FullName, Length
Get-FileHash -Algorithm SHA256 -LiteralPath $finalPath | Select-Object Hash
