[CmdletBinding()]
param(
    [string]$OutputPath = "MWTopLookups.csv",
    [int]$IntervalSeconds = 31,
    [int]$MaxPolls = 0,
    [switch]$RunOnce,
    [int]$RequestTimeoutSeconds = 20,
    [int]$RetryCount = 3,
    [int]$RetryDelaySeconds = 2,
    [string]$Endpoint = "https://factotum196p963.m-w.com:6058/lapi/v1/mwol-mp/get-lookups-data-homepage"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-StringSha256 {
    param([Parameter(Mandatory = $true)][string]$Value)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
        $hashBytes = $sha.ComputeHash($bytes)
        return ([System.BitConverter]::ToString($hashBytes) -replace '-', '').ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Get-MwTopLookups {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)][int]$Attempts,
        [Parameter(Mandatory = $true)][int]$DelaySeconds
    )

    $headers = @{
        "User-Agent" = "Mozilla/5.0"
        "Referer"    = "https://www.merriam-webster.com/"
        "Origin"     = "https://www.merriam-webster.com"
        "Accept"     = "application/json, text/plain, */*"
    }

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            $response = Invoke-RestMethod -Uri $Uri -Headers $headers -Method Get -TimeoutSec $TimeoutSeconds
            if ($null -eq $response.data -or $null -eq $response.data.words) {
                throw "Response missing data.words"
            }
            return $response.data
        }
        catch {
            if ($attempt -eq $Attempts) {
                throw
            }
            Start-Sleep -Seconds $DelaySeconds
        }
    }
}

function Write-LookupRows {
    param(
        [Parameter(Mandatory = $true)]$LookupData,
        [Parameter(Mandatory = $true)][string]$CsvPath,
        [Parameter(Mandatory = $true)][int]$PollIndex,
        [Parameter(Mandatory = $true)][string]$Uri
    )

    $polledAtLocal = Get-Date
    $polledAtUtc = $polledAtLocal.ToUniversalTime()
    $pollId = "{0}-{1}" -f $polledAtUtc.ToString("yyyyMMddTHHmmss.fffZ"), ([Guid]::NewGuid().ToString("N").Substring(0, 8))

    $words = @($LookupData.words)
    $joined = [string]::Join("|", $words)
    $listHash = Get-StringSha256 -Value $joined

    $rows = @()
    for ($i = 0; $i -lt $words.Count; $i++) {
        $word = [string]$words[$i]
        $rows += [PSCustomObject]@{
            poll_id           = $pollId
            poll_index        = $PollIndex
            polled_at_utc     = $polledAtUtc.ToString("o")
            polled_at_local   = $polledAtLocal.ToString("o")
            source_timestamp  = [string]$LookupData.timestamp
            rank              = $i + 1
            word              = $word
            word_normalized   = $word.Trim().ToLowerInvariant()
            list_hash         = $listHash
            endpoint          = $Uri
        }
    }

    if ($rows.Count -eq 0) {
        Write-Warning "No words returned for poll_index=$PollIndex"
        return
    }

    $outputDir = Split-Path -Path $CsvPath -Parent
    if ($outputDir -and -not (Test-Path -LiteralPath $outputDir)) {
        New-Item -Path $outputDir -ItemType Directory -Force | Out-Null
    }

    $append = Test-Path -LiteralPath $CsvPath
    $rows | Export-Csv -Path $CsvPath -NoTypeInformation -Encoding UTF8 -Append:$append

    $top3 = ($rows | Select-Object -First 3 | ForEach-Object { "#{0} {1}" -f $_.rank, $_.word }) -join "; "
    Write-Host ("[{0}] Logged {1} rows to {2}. Top: {3}" -f $polledAtLocal.ToString("yyyy-MM-dd HH:mm:ss"), $rows.Count, $CsvPath, $top3)
}

if ($IntervalSeconds -lt 5) {
    throw "IntervalSeconds must be at least 5"
}
if ($MaxPolls -lt 0) {
    throw "MaxPolls cannot be negative"
}
if ($RetryCount -lt 1) {
    throw "RetryCount must be at least 1"
}
if ($RetryDelaySeconds -lt 1) {
    throw "RetryDelaySeconds must be at least 1"
}

$pollIndex = 0
$targetPolls = if ($RunOnce) { 1 } elseif ($MaxPolls -gt 0) { $MaxPolls } else { [int]::MaxValue }

while ($pollIndex -lt $targetPolls) {
    try {
        $lookupData = Get-MwTopLookups -Uri $Endpoint -TimeoutSeconds $RequestTimeoutSeconds -Attempts $RetryCount -DelaySeconds $RetryDelaySeconds
        $pollIndex++
        Write-LookupRows -LookupData $lookupData -CsvPath $OutputPath -PollIndex $pollIndex -Uri $Endpoint
    }
    catch {
        Write-Warning ("Poll failed: {0}" -f $_.Exception.Message)
        if ($RunOnce) {
            throw
        }
    }

    if ($pollIndex -ge $targetPolls) {
        break
    }

    Start-Sleep -Seconds $IntervalSeconds
}
