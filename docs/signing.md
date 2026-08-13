# Code signing & notarization (release checklist)

This document describes what is required to ship **signed** UAII installers on
Windows and macOS. **UAII does not currently distribute signed binaries** —
community and CI artifacts are **unsigned** unless an organization attaches its
own certificates in a private release pipeline.

## Why signing needs org certs

| Platform | Typical credential | Who issues it |
|---|---|---|
| Windows | Authenticode code-signing certificate (OV/EV) | Public CA via your org |
| macOS | Apple Developer ID Application certificate + notarization | Apple Developer Program (org/team) |

Personal or self-signed certificates do **not** satisfy SmartScreen / Gatekeeper
expectations for public downloads. Do not claim a release is “signed” unless
these steps completed successfully with org-owned certs.

## Windows checklist (`signtool`)

1. Obtain an Authenticode cert (`.pfx` or hardware token) owned by your org.
2. Install [Windows SDK](https://developer.microsoft.com/windows/downloads/windows-sdk/) so `signtool.exe` is on `PATH`.
3. After building `uaii.exe` (and any DLLs you ship):

```bat
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /f org.pfx /p <PASSWORD> uaii.exe
signtool verify /pa uaii.exe
```

4. Prefer EV + timestamping so signatures remain valid after cert expiry.
5. Package the signed binary with `LICENSE` + `README` (see unsigned helper below).

## macOS checklist (codesign + notarize)

1. Enroll in the Apple Developer Program; create a **Developer ID Application** cert.
2. Sign the Mach-O binary / `.app` / `.dmg`:

```bash
codesign --force --options runtime --sign "Developer ID Application: Your Org" ./uaii
codesign --verify --verbose ./uaii
```

3. Notarize (example with `notarytool`):

```bash
xcrun notarytool submit uaii.zip --apple-id <ID> --team-id <TEAM> --password <APP_SPECIFIC> --wait
xcrun stapler staple ./uaii   # if applicable to the artifact type
```

4. Gatekeeper-check on a clean Mac before publishing.

## Unsigned packaging helper

For local / CI zips **without** signing, use:

```powershell
powershell -File scripts/package_release.ps1
```

That script archives `uaii.exe` (or `uaii`) plus `LICENSE` and `README.md`. It
does **not** invoke `signtool` or Apple notarization.

## Honesty

- README / release notes should say **unsigned** unless org signing succeeded.
- Do not embed fake trust badges or “verified publisher” claims for CI artifacts.
