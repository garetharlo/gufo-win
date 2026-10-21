# start-gufo-flash-next.ps1 - bosgame launcher for the pixmaate/gufo Windows port.
# Wraps tools\windows\run-flash-next-xl.ps1 via the C:\gufo-win junction
# so every path the engine sees is SPACE-FREE (the build/compile system chokes on
# paths with spaces; see workspace AGENTS.md round 40).
#
#   start-gufo-flash-next.bat                          double-click = serve (thinking on, adaptive MTP)
#   powershell -File .\start-gufo-flash-next.ps1 -Think off    instruct mode (temp 0.7/top-p 0.8)
#   ... -Draft mtp3                                    MTP capped at 3 drafts
#   ... -Draft off                                     autoregressive baseline
#   ... -Mode bench                                    gufo's own pp2048/tg128 depth table (0..128K)
#   ... -Survival off -DraftVocab off                   disable the defaults (now ON:
#                                                      Reddit post's headline config; latin draft
#                                                      vocab = ENGLISH/CODE ONLY - non-Latin just
#                                                      loses the MTP speedup, quality unchanged)
#   ... -Context 131072                                smaller ctx if 256K does not fit
#   ... -ServedName <id>                               served model id (default qwen3.8-flash-next,
#                                                      = OpenCode's existing model key; gufo VALIDATES
#                                                      model strings, llama-server ignored them)
#   ... -BindHost <h>                                  0.0.0.0 = accept connections from anywhere (the
#                                                      default, LAN-exposed like the llama engine - its
#                                                      firewall rule for TCP 8888 already exists);
#                                                      127.0.0.1 = loopback-only
#   ... -LoadExpectSec <s>                             expected load time for the % ESTIMATE in the
#                                                      loading heartbeat (default 80 s: measured loads
#                                                      run 35-41 s warm / 61-79 s cold; gufo exposes no
#                                                      real progress - % is elapsed/expected, capped 99)
#   ... -DryRun                                        preflight + command plan, no start
#
# Default port is 8888 on this box (same as the llama engine: firewall rule + client
# configs already exist, and only one engine can serve at a time anyway).
# 8080 = the user's Open WebUI - never touch it.
#
# Model files auto-resolve from the HF cache (UD-Q4_K_XL trunk + MTP shared-Q8_0
# sidecar - already on disk). Server binds 127.0.0.1; the launcher POLLS /health
# and prints a green "MODEL LOADED" banner (+ beep) the moment it returns 200 -
# gufo's own console drawing/event lines do not survive the tee, so rely on that.
#
# SAFETY: gufo needs ~89 GiB of the 96 GB carve at 256K ctx - the llama.cpp
# engine MUST be stopped first. This launcher refuses to start alongside a
# running llama-server unless you pass -Force. It NEVER kills engines itself.
param(
  [ValidateSet("serve", "bench")] [string]$Mode = "serve",
  [ValidateSet("on", "off")] [string]$Think = "on",
  # Server default for how much the model reasons (gufo --reasoning-effort);
  # "auto" = the model's own default and adds no flag. Dropped when -Think is
  # off: gufo rejects an explicit effort in that mode.
  [ValidateSet("auto", "minimal", "low", "medium", "high", "xhigh", "max")] [string]$ReasoningEffort = "auto",
  [ValidateSet("mtp", "mtp3", "off")] [string]$Draft = "mtp",
  [switch]$Greedy,
  # Survival + latin draft vocab = the Reddit post's headline config: ON by default
  # (repo-measured: latin +4.9% decode, survival +8.7% on top; post's 63-78%
  # acceptance came WITH both). Opt OUT with -Survival off / -DraftVocab off
  # (ValidateSet strings, NOT switches: -X:$false breaks under powershell -File).
  # Latin drafts ENGLISH/CODE ONLY - quality is unaffected, but non-Latin text
  # just loses the MTP speedup.
  [ValidateSet("on","off")] [string]$DraftVocab = "on",
  [ValidateSet("on","off")] [string]$Survival   = "on",
  [switch]$Lookup,
# [int]$Context = 262144,
  [int]$Context = 131072,
  # Session pool (passes through to run-flash-next-xl.ps1; reservation is
  # sessions*context of state - see that script). Default 1 = unchanged.
  [int]$Sessions = 1,
  # Per-client-IP pending-request cap passed to the run script. Every local
  # client is 127.0.0.1, so local multi-agent serving needs this raised or the
  # 5th concurrent request is rejected (429 client_queue_full).
  [int]$MaxPendingPerClient = 4,
  [int]$Port = 8888,   # 8888: same port as the llama engine - its firewall rule and client
                       # configs already point here, and the two engines can never run at the
                       # same time (one 96 GB carve). 8080 = user's Open WebUI - do not touch.
  # Expected load duration (seconds) - only used for the heartbeat's % estimate;
  # 40 s covers the observed 35-38 s full loads of Q4_K_XL + MTP without the
  # percentage pegging at 99 for the last stretch.
  [ValidateRange(1, 3600)] [int]$LoadExpectSec = 80,
  [string]$Snapshot = "",
  [string]$MtpModel = "",
  # Vision: --mmproj (mmproj-F16.gguf, auto-resolved in the HF snapshots).
  # ON by default; -NoVision = text-only profile.
  [switch]$NoVision,
  # Bind host: 0.0.0.0 = accept connections from anywhere (LAN-exposed, same
  # stance as the llama engine round-25b; the TCP 8888 firewall rule exists).
  [string]$BindHost = "0.0.0.0",
  # gufo rejects requests whose model string != --served-model-name; default matches
  # the existing OpenCode config entry so the SAME model id works with both engines.
  [string]$ServedName = "unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL",
  # Serve with the gufo-sync build (C:\gufo-sync) instead of gufo-win.
  [switch]$Sync,
  [switch]$Force,
  [switch]$DryRun
)
$ErrorActionPreference = "Stop"

$Junction  = if ($Sync) { "C:\gufo-sync" } else { "C:\gufo-win" }
$GufoTree  = "C:\dev\gufo-win"
$SyncTree  = "C:\dev\gufo-sync"
$ThisDir   = $PSScriptRoot
$RunScript = Join-Path $Junction "tools\windows\run-flash-next-xl.ps1"
$GufoExe   = Join-Path $Junction "build\release\gufo.exe"
$Log       = Join-Path $ThisDir "gufo-server-log.txt"
Write-Host "=== gufo launcher (round 40) ===" -ForegroundColor Cyan

# 1. Junction alias so the engine only ever sees space-free paths.
if (-not (Test-Path $Junction)) {
  Write-Host "creating junction ${Junction} -> $(if ($Sync) { $SyncTree } else { $GufoTree })"
  New-Item -ItemType Junction -Path $Junction -Target $(if ($Sync) { $SyncTree } else { $GufoTree }) | Out-Null
}
if (-not (Test-Path $RunScript)) { throw "run script not found: ${RunScript}" }

# 2. Built binary present?
$buildOk = Test-Path $GufoExe
if ($buildOk) {
  Write-Host "binary  : $GufoExe"
} else {
  Write-Host "binary  : MISSING ($GufoExe)" -ForegroundColor Yellow
  Write-Host "  build first: powershell -ExecutionPolicy Bypass -File ${Junction}\tools\windows\build.ps1"
}

# 3. Port free?
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
$portFree = (-not $listener)
if ($portFree) {
  Write-Host "port    : ${Port} free"
} else {
  Write-Host "port    : ${Port} IN USE (pid $($listener[0].OwningProcess))" -ForegroundColor Yellow
}

# 3b. LAN address for the ready banner - resolved BEFORE gufo starts (pure DNS
#     API, no WMI/CIM: those stall under load pressure; see AGENTS.md round 30c).
$LanIp = $null
if ($BindHost -eq "0.0.0.0") {
  try {
    $ips = [System.Net.Dns]::GetHostAddresses($env:COMPUTERNAME) |
      Where-Object { $_.AddressFamily -eq 'InterNetwork' -and $_.IPAddressToString -notmatch '^(127\.|169\.254\.)' }
    $pick = $ips | Where-Object { $_.IPAddressToString -like '192.168.*' } | Select-Object -First 1
    if (-not $pick) { $pick = $ips | Select-Object -First 1 }
    if ($pick) { $LanIp = $pick.IPAddressToString }
  } catch { }
  if ($LanIp) { Write-Host "bind    : ${BindHost} (LAN http://${LanIp}:${Port}/v1)" }
}

# 4. The 96 GB carve is exclusive - refuse to run next to our llama.cpp engine.
$llama = Get-Process llama-server -ErrorAction SilentlyContinue
if ($llama) {
  if ($Force) {
    Write-Host "carve   : llama-server running (pid $($llama.Id -join ', ')) but -Force given" -ForegroundColor Yellow
  } else {
    Write-Host "carve   : llama-server RUNNING (pid $($llama.Id -join ', ')) - two engines cannot share the 96 GB carve" -ForegroundColor Red
    Write-Host "  stop it first (it is YOUR engine; this launcher never kills anything), or pass -Force" -ForegroundColor Red
  }
} else {
  Write-Host "carve   : no llama-server running - good"
}

# 5. Build the inner invocation (same params, plus the opt-in MTP switches).
$inner = @($RunScript, "-Mode", $Mode, "-Think", $Think, "-Draft", $Draft,
           "-Context", "$Context", "-Sessions", "$Sessions",
           "-MaxPendingPerClient", "$MaxPendingPerClient",
           "-Port", "$Port", "-BindHost", $BindHost,
           "-ServedName", $ServedName)
if ($ReasoningEffort -ne "auto" -and $Think -eq "on") { $inner += @("-ReasoningEffort", $ReasoningEffort) }
if ($Greedy)     { $inner += "-Greedy" }
if ($DraftVocab -eq "on") { $inner += "-DraftVocab" }
if ($Survival   -eq "on") { $inner += "-Survival" }
if ($Lookup)     { $inner += "-Lookup" }
if ($NoVision)   { $inner += "-NoVision" }
if ($Snapshot)   { $inner += @("-Snapshot", $Snapshot) }
if ($MtpModel)   { $inner += @("-MtpModel", $MtpModel) }

$dv = $DraftVocab   # 'on'/'off' strings - echo exactly what was bound
$sv = $Survival
Write-Host "mode    : ${Mode}  think=${Think}  draft=${Draft}  ctx=${Context}  port=${Port}  host=${BindHost}  id=${ServedName}"
Write-Host "drafting: survival=${sv}  draft-vocab=${dv}  (lookup opt-in; latin = English/code only)"
Write-Host "log     : ${Log}"
Write-Host "command : powershell -File $($inner -join ' ')"

if ($DryRun) {
  Write-Host "`nDryRun - nothing started." -ForegroundColor Cyan
  exit 0
}

if (-not $buildOk) { throw "gufo.exe not built yet - cannot start (see path above)." }
if (-not $portFree) { throw "port ${Port} is already in use - pick another with -Port <n>." }
if ($llama -and -not $Force) { throw "refusing to start: llama-server still running (pass -Force to override)." }

# 6. Run gufo with output teed to gufo-server-log.txt (APPENDED, so load logs
#    from previous runs survive binary swaps) in a
#    background pipeline, while the MAIN thread polls /health and prints a
#    MODEL LOADED banner in the console the moment the server is ready
#    (gufo's own event=load_completed lines do not survive the tee).
#    Ctrl+C stops everything (one console process group).
Write-Host "`nstarting gufo (log: ${Log}) ..." -ForegroundColor Cyan

$env:GUFO_EXIT = '0'
$runBlock = {
  param($RunScript2, $InnerArgs, $LogFile)
  & powershell -NoProfile -ExecutionPolicy Bypass -File @InnerArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
  [Environment]::SetEnvironmentVariable('GUFO_EXIT', "$LASTEXITCODE", 'Process')
}
$ps = [powershell]::Create()
$null = $ps.AddScript($runBlock).AddArgument($RunScript).AddArgument($inner).AddArgument($Log)
$handle = $ps.BeginInvoke()

$ready = $false
if ($Mode -eq 'serve') {
  $t0 = Get-Date
  Write-Host "waiting for the model to load (polling http://127.0.0.1:${Port}/health) ..."
  $pad = 0
  while (-not $handle.IsCompleted) {
    Start-Sleep -Seconds 1
    try {
      $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -UseBasicParsing
      if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    $elapsed = (Get-Date) - $t0
    $pct = [Math]::Min(99, [int](100 * $elapsed.TotalSeconds / $LoadExpectSec))
    $msg = ("  loading ... {0,3}% ({1:mm\:ss})" -f $pct, $elapsed)
    $clr = ' ' * [Math]::Max(0, $pad - $msg.Length)
    Write-Host "`r$msg$clr" -NoNewline
    $pad = $msg.Length
  }
  Write-Host ""   # end the heartbeat line
  if ($ready) {
    $loadSec = [int]((Get-Date) - $t0).TotalSeconds
    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host " MODEL LOADED in ${loadSec}s - http://127.0.0.1:${Port}/v1" -ForegroundColor Green
    if ($LanIp) {
      Write-Host " LAN: http://${LanIp}:${Port}/v1   model id: ${ServedName}   (Ctrl+C to stop)" -ForegroundColor Green
    } else {
      Write-Host " model id: ${ServedName}   (Ctrl+C to stop)" -ForegroundColor Green
    }
    Write-Host "========================================================" -ForegroundColor Green
    try { [console]::beep(880,200); [console]::beep(1100,300) } catch { }
  } else {
    Write-Host "server exited before /health went ok - check ${Log}" -ForegroundColor Red
  }
}

# Block until gufo exits (Ctrl+C), then report its exit code.
try { $null = $ps.EndInvoke($handle) } catch { Write-Host "gufo pipeline error: $_" -ForegroundColor Red }
$ps.Dispose()
$code = [int]$env:GUFO_EXIT
Write-Host "`ngufo exited with code ${code}." -ForegroundColor $(if ($code -eq 0) { "Cyan" } else { "Red" })
exit $code
