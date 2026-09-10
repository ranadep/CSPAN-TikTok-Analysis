#!/usr/bin/env bash
set -euo pipefail

# Download videos from the official C-SPAN TikTok account.
#
# Default project layout:
#
#   c-span-videos/
#     download_tiktok_cspan.sh
#     videos/
#     video-manifest.tsv
#
# The manifest is a tab-separated file with:
#
#   source_url    downloaded_file_path

# By default, download from the C-SPAN TikTok profile.
# You can pass a different profile/video URL as the first argument.
ACCOUNT_URL="${1:-https://www.tiktok.com/@cspanofficial}"

# Always run relative to the folder where this script lives.
# This keeps videos/ and video-manifest.tsv in the expected project folder
# even if you start the script from another terminal location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional batch mode:
# If video_urls.txt exists, the script downloads URLs from that file instead
# of crawling the TikTok profile page. Put one video URL per line.
URLS_FILE="${URLS_FILE:-video_urls.txt}"

# Where downloaded videos are saved.
# Default: ./videos
# Example external-drive usage:
#   VIDEO_DIR="/Volumes/MyDrive/cspan-videos/videos" ./download_tiktok_cspan.sh
VIDEO_DIR="${VIDEO_DIR:-videos}"

# Be gentle with the site: wait between requests and downloads.
# Defaults:
#   REQUEST_SLEEP=3  waits 3 seconds between smaller web requests
#   SLEEP_MIN=20     minimum wait between video downloads
#   SLEEP_MAX=60     maximum wait between video downloads
#
# Slower example:
#   SLEEP_MIN=60 SLEEP_MAX=180 REQUEST_SLEEP=5 ./download_tiktok_cspan.sh
SLEEP_MIN="${SLEEP_MIN:-20}"
SLEEP_MAX="${SLEEP_MAX:-60}"
REQUEST_SLEEP="${REQUEST_SLEEP:-3}"

# Optional small test run.
# Example:
#   MAX_DOWNLOADS=3 ./download_tiktok_cspan.sh
MAX_DOWNLOADS="${MAX_DOWNLOADS:-}"

# Only use this if you see SSL/certificate handshake errors.
# We needed this in Anvil/Jupyter, but the local Mac test worked without it.
# Example:
#   SSL_WORKAROUND=1 MAX_DOWNLOADS=3 ./download_tiktok_cspan.sh
SSL_WORKAROUND="${SSL_WORKAROUND:-0}"

cd "$SCRIPT_DIR"

# Create the video output folder if it does not exist yet.
mkdir -p "$VIDEO_DIR"

# Prefer the yt-dlp command if it is on PATH.
# If not, use the Python module form. This is useful after installing with:
#   python3 -m pip install --user -U "yt-dlp[default,curl-cffi]"
if command -v yt-dlp >/dev/null 2>&1; then
  YTDLP=(yt-dlp)
else
  YTDLP=(python3 -m yt_dlp)
fi

# Build optional yt-dlp arguments.
EXTRA_ARGS=()
if [[ -n "$MAX_DOWNLOADS" ]]; then
  EXTRA_ARGS+=(--playlist-end "$MAX_DOWNLOADS")
fi

if [[ "$SSL_WORKAROUND" == "1" ]]; then
  EXTRA_ARGS+=(--no-check-certificate --legacy-server-connect)
fi

# Input source:
# - If video_urls.txt exists, download those exact URLs.
# - Otherwise, crawl the C-SPAN TikTok profile URL.
INPUT_ARGS=("$ACCOUNT_URL")
if [[ -f "$URLS_FILE" ]]; then
  INPUT_ARGS=(--batch-file "$URLS_FILE")
fi

# Main yt-dlp command.
#
# Important outputs:
# - Videos:              $VIDEO_DIR/
# - Video metadata:      $VIDEO_DIR/*.info.json
# - URL/file manifest:   video-manifest.tsv
# - Download archive:    .downloaded-archive.txt
#
# .downloaded-archive.txt lets you rerun this script later without
# redownloading videos that already finished successfully.
"${YTDLP[@]}" \
  --yes-playlist \
  --ignore-errors \
  --continue \
  --no-overwrites \
  --download-archive .downloaded-archive.txt \
  --sleep-requests "$REQUEST_SLEEP" \
  --sleep-interval "$SLEEP_MIN" \
  --max-sleep-interval "$SLEEP_MAX" \
  --restrict-filenames \
  --merge-output-format mp4 \
  --write-info-json \
  -o "$VIDEO_DIR/%(upload_date>%Y-%m-%d)s_%(id)s.%(ext)s" \
  --print-to-file "after_move:%(webpage_url)s	%(filepath)s" video-manifest.tsv \
  "${EXTRA_ARGS[@]}" \
  "${INPUT_ARGS[@]}"

# ---------------------------------------------------------------------------
# How to run this file next time
# ---------------------------------------------------------------------------
#
# 1. Go to this project folder:
#
#      cd /Users/jingying/Desktop/c-span-videos
#
# 2. Install/update yt-dlp if needed:
#
#      python3 -m pip install --user -U "yt-dlp[default,curl-cffi]"
#
# 3. Make the script executable:
#
#      chmod +x download_tiktok_cspan.sh
#
# 4. Test with 3 videos first:
#
#      MAX_DOWNLOADS=3 ./download_tiktok_cspan.sh
#
# 5. If the test looks good, run the slower full download:
#
#      SLEEP_MIN=60 SLEEP_MAX=180 REQUEST_SLEEP=5 ./download_tiktok_cspan.sh
#
# 6. To save videos to an external drive:
#
#      SLEEP_MIN=60 SLEEP_MAX=180 REQUEST_SLEEP=5 VIDEO_DIR="/Volumes/MyDrive/cspan-videos/videos" ./download_tiktok_cspan.sh
#
# 7. To download a hand-picked list of videos:
#
#      Create video_urls.txt in this folder, one TikTok video URL per line,
#      then run:
#
#      SLEEP_MIN=60 SLEEP_MAX=180 REQUEST_SLEEP=5 ./download_tiktok_cspan.sh
