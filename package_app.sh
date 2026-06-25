#!/usr/bin/env bash
# Build a lightweight macOS .app wrapper for this checkout.

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="Voice Transcribe Qwen"
OUT_DIR="${1:-${APP_DIR}/dist}"
APP_PATH="${OUT_DIR}/${APP_NAME}.app"
CONTENTS="${APP_PATH}/Contents"
MACOS="${CONTENTS}/MacOS"

mkdir -p "${MACOS}" "${CONTENTS}/Resources"
rm -f "${MACOS}/VoiceTranscribeQwen"

cat > "${CONTENTS}/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>Voice Transcribe Qwen</string>
  <key>CFBundleExecutable</key>
  <string>VoiceTranscribeQwen</string>
  <key>CFBundleIdentifier</key>
  <string>com.local.voice-transcribe-qwen</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>Voice Transcribe Qwen</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>Voice Transcribe Qwen records speech locally while you hold Fn.</string>
</dict>
</plist>
PLIST

cat > "${MACOS}/VoiceTranscribeQwen" <<EOF
#!/usr/bin/env bash
cd "${APP_DIR}"
exec "${APP_DIR}/run.sh" >> /tmp/voice-transcribe.log 2>&1
EOF
chmod +x "${MACOS}/VoiceTranscribeQwen"

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "${APP_PATH}" >/dev/null 2>&1 || true
fi

echo "Created ${APP_PATH}"
