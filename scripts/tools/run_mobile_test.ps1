<#
รวมทุกขั้นตอนทดสอบ Web Push บนมือถือจริงผ่าน LAN ให้เหลือคำสั่งเดียว:
หา LAN IP -> สร้าง/อัปเดต dev cert ให้ตรง IP -> build frontend -> ตั้ง HTTPS/LAN -> รัน Flask

ใช้:
    .\scripts\tools\run_mobile_test.ps1            (หา LAN IP ให้อัตโนมัติ)
    .\scripts\tools\run_mobile_test.ps1 10.1.2.3   (ระบุ IP เอง ถ้า auto-detect ได้ IP ผิด)

กด Ctrl+C เพื่อหยุด server -- env var HTTPS/LAN จะหายไปเองเมื่อปิดหน้าต่าง PowerShell นี้
(ไม่ค้างไปกระทบเทอร์มินัลอื่นหรือ terminal ที่เปิดใหม่)
#>
param(
    [string]$IpOverride
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # scripts\tools -> root ของโปรเจกต์
Set-Location $root

if ($IpOverride) {
    $ip = $IpOverride
} else {
    $ip = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike "169.254.*" -and $_.IPAddress -ne "127.0.0.1" } |
        Where-Object { $_.InterfaceAlias -notmatch "Loopback|vEthernet|VMware|VirtualBox|Tailscale|WSL" } |
        Select-Object -First 1 -ExpandProperty IPAddress)
}

if (-not $ip) {
    Write-Error "หา LAN IP ไม่เจอ (เช็คว่าต่อ Wi-Fi/Ethernet อยู่ไหม) ลองระบุเอง: .\scripts\tools\run_mobile_test.ps1 <IP>"
    exit 1
}

Write-Host "LAN IP: $ip" -ForegroundColor Cyan

python scripts/tools/generate_dev_cert.py $ip

Push-Location frontend
npm run build
Pop-Location

$env:HTTPS = "1"
$env:LAN = "1"

$url = "https://" + $ip + ":5000"
Write-Host ""
Write-Host ("เปิดจากมือถือ (ต้องอยู่ Wi-Fi วงเดียวกับ PC): " + $url) -ForegroundColor Green
Write-Host "iOS ต้อง Add to Home Screen แล้วเปิดจากไอคอนนั้นก่อนถึงจะรับ push ได้" -ForegroundColor Yellow
Write-Host ""

python app.py
