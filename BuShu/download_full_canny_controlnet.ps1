$ErrorActionPreference = "Stop"

$downloadRoot = Join-Path $env:USERPROFILE "Downloads\controlnet-canny-sdxl-1.0"
$partialPath = Join-Path $downloadRoot "diffusion_pytorch_model.fp16.safetensors.part"
$finalPath = Join-Path $downloadRoot "diffusion_pytorch_model.fp16.safetensors"
$configPath = Join-Path $downloadRoot "config.json"
$weightUrl = "https://hf-mirror.com/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors"
$configUrl = "https://hf-mirror.com/diffusers/controlnet-canny-sdxl-1.0/resolve/main/config.json"
$expectedSize = 2502139136
$expectedHash = "b2e7d3921058a442cc80430d1ec8847f42599c705e2451c95e77cf4dcf8d6c25"

New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $finalPath)) {
    Write-Host "开始或继续下载完整版 SDXL Canny ControlNet。"
    & curl.exe -L --fail --connect-timeout 30 --speed-time 120 --speed-limit 1024 `
        -C - -o $partialPath $weightUrl
    if ($LASTEXITCODE -ne 0) {
        Write-Host "连接已中断，断点文件保留在：$partialPath"
        Write-Host "重新运行同一脚本即可继续，请勿删除 .part 文件。"
        exit $LASTEXITCODE
    }

    $downloadedSize = (Get-Item -LiteralPath $partialPath).Length
    if ($downloadedSize -ne $expectedSize) {
        throw "文件大小异常：$downloadedSize，预期：$expectedSize"
    }
    $downloadedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $partialPath).Hash.ToLowerInvariant()
    if ($downloadedHash -ne $expectedHash) {
        $quarantinePath = $partialPath + ".corrupt-" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $partialPath -Destination $quarantinePath
        throw "SHA-256 不匹配；损坏文件已隔离到：$quarantinePath"
    }
    Move-Item -LiteralPath $partialPath -Destination $finalPath -Force
}

$currentHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $finalPath).Hash.ToLowerInvariant()
if ($currentHash -ne $expectedHash) {
    $quarantinePath = $finalPath + ".corrupt-" + (Get-Date -Format "yyyyMMdd_HHmmss")
    Move-Item -LiteralPath $finalPath -Destination $quarantinePath
    throw "现有正式权重 SHA-256 不匹配，已隔离到：$quarantinePath"
}

& curl.exe -L --fail --connect-timeout 30 -o $configPath $configUrl
if ($LASTEXITCODE -ne 0) {
    throw "权重已完成，但 config.json 下载失败；重新运行本脚本即可。"
}

Write-Host "下载与校验完成。请把下面两个文件复制到项目模型目录："
Write-Host "E:\App\DaiMa\JetBrains\Project\TuXiangAI\models\controlnet-canny-sdxl-1.0"
Get-Item -LiteralPath $finalPath, $configPath | Select-Object FullName, Length
Get-FileHash -Algorithm SHA256 -LiteralPath $finalPath | Select-Object Hash
