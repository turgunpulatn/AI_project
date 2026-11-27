<#
Share VoxCheck with classmates through a temporary public HTTPS link (Cloudflare quick tunnel).

  powershell -ExecutionPolicy Bypass -File scripts\share.ps1            # new random access code
  powershell -ExecutionPolicy Bypass -File scripts\share.ps1 -Code 4821 # your own code

- Needs cloudflared once:  winget install --id Cloudflare.cloudflared
- No domain or account needed. The link works only while this window is open and the laptop is on.
- HTTPS matters: browsers only allow the microphone on https:// (or localhost).
- Saving recordings requires the access code; recordings land in dataset/ on THIS laptop,
  with a line in dataset/consent_log.csv for each one.
- Press Ctrl+C to stop both the app and the link.
#>
param(
    [string]$Code = "",
    [int]$Port = 5050
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$cf = Join-Path $root "tools\cloudflared.exe"           # portable copy (no admin needed)
if (-not (Test-Path $cf)) {
    $onPath = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($onPath) { $cf = $onPath.Source }
    else {
        Write-Host ""
        Write-Host "cloudflared not found. Either install it:  winget install --id Cloudflare.cloudflared" -ForegroundColor Yellow
        Write-Host "or download cloudflared-windows-amd64.exe from https://github.com/cloudflare/cloudflared/releases"
        Write-Host "and save it as tools\cloudflared.exe (no admin needed)."
        exit 1
    }
}

if (-not $Code) { $Code = -join ((1..6) | ForEach-Object { "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"[(Get-Random -Maximum 32)] }) }
$env:VOX_ACCESS_CODE = $Code
$log = Join-Path $env:TEMP "voxcheck_tunnel.log"
Remove-Item $log -ErrorAction SilentlyContinue

Write-Host "Starting VoxCheck on port $Port (loading the model takes ~20 s)..."
$app = Start-Process python -ArgumentList "app.py", "--port", "$Port" -PassThru -NoNewWindow
$tunnel = $null
try {
    $ok = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        if ($app.HasExited) { throw "VoxCheck stopped while starting - see the messages above." }
        try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($h.ok) { $ok = $true; break } } catch {}
    }
    if (-not $ok) { throw "VoxCheck did not become ready on port $Port." }

    Write-Host "Opening the public HTTPS link..."
    # http2 = plain HTTPS over TCP 443: works on networks that block UDP/QUIC (the default protocol)
    $tunnel = Start-Process $cf -ArgumentList "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", "http://127.0.0.1:$Port" `
        -PassThru -NoNewWindow -RedirectStandardError $log
    $url = $null; $connected = $false
    for ($i = 0; $i -lt 90 -and -not $connected; $i++) {
        Start-Sleep -Seconds 1
        if ($tunnel.HasExited) { throw "cloudflared stopped - see $log" }
        if (Test-Path $log) {
            if (-not $url) {
                $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
                if ($m) { $url = $m.Matches[0].Value }
            }
            # the link only works once a connection to Cloudflare is registered
            if ($url -and (Select-String -Path $log -Pattern "Registered tunnel connection" -Quiet)) { $connected = $true }
        }
    }
    if (-not $connected) { throw "The tunnel did not connect (network blocks Cloudflare?) - see $log" }
    for ($i = 0; $i -lt 30; $i++) {            # wait until the public link really answers
        try { $h = Invoke-RestMethod "$url/health" -TimeoutSec 5; if ($h.ok) { break } } catch {}
        Start-Sleep -Seconds 2
    }

    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host "  Send this to your classmates:" -ForegroundColor Green
    Write-Host "    Link:        $url" -ForegroundColor Green
    Write-Host "    Access code: $Code" -ForegroundColor Green
    Write-Host "  They open the link -> 'Collect data' tab -> enter the code,"
    Write-Host "  tick the consent box, record, save."
    Write-Host "  Keep this window open. Ctrl+C stops the link and the app."
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host ""
    Wait-Process -Id $app.Id
}
finally {
    foreach ($p in @($tunnel, $app)) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
    Write-Host "Stopped the link and the app."
}
