#!/bin/bash
# Install a named macOS app wrapper so permissions show "Qwen Dictate" instead of "Python".

set -euo pipefail

APP_NAME="Qwen Dictate"
BUNDLE_ID="com.zackseyun.qwen-dictate"
PYTHON_APP="/opt/homebrew/Cellar/python@3.13/3.13.1/Frameworks/Python.framework/Versions/3.13/Resources/Python.app"
TARGET_APP="/Applications/${APP_NAME}.app"
ICON_SOURCE="/System/Library/PrivateFrameworks/SpeechObjects.framework/Versions/A/SpeechDataInstallerd.app/Contents/Resources/Dictation.icns"
ICON_NAME="VoiceTranscribe.icns"

if [[ ! -d "$PYTHON_APP" ]]; then
  echo "Python.app not found at: $PYTHON_APP" >&2
  exit 1
fi

if [[ -e "$TARGET_APP" ]]; then
  rm -rf "$TARGET_APP"
fi

cp -R "$PYTHON_APP" "$TARGET_APP"
cp "$ICON_SOURCE" "$TARGET_APP/Contents/Resources/$ICON_NAME"

/usr/libexec/PlistBuddy \
  -c "Set :CFBundleName $APP_NAME" \
  -c "Delete :CFBundleDisplayName" \
  "$TARGET_APP/Contents/Info.plist" 2>/dev/null || true

/usr/libexec/PlistBuddy \
  -c "Set :CFBundleName $APP_NAME" \
  -c "Add :CFBundleDisplayName string $APP_NAME" \
  -c "Set :CFBundleIdentifier $BUNDLE_ID" \
  -c "Set :CFBundleIconFile $ICON_NAME" \
  "$TARGET_APP/Contents/Info.plist"

echo "Installed $TARGET_APP"
