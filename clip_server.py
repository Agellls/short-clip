# clip_server.py
"""
Clip + Caption server (updated)

Changes:
 - Accept optional `resolution` (e.g. "1080x1920") in requests for /clip and /clip_with_caption
 - Resize/reencode output to requested resolution (scale+pad to preserve aspect ratio)
 - Resize before burning/embedding subtitles so final output matches requested size
 - Accept optional background music URL and settings in /clip_with_caption
 - Download and add background music to video, with volume and fade control

Usage example request JSON:
{
  "url":"https://www.youtube.com/watch?v=v52S3LBFZJs",
  "start":10,
  "end":40,
  "burn":true,
  "resolution":"1080x1920",
  "words_per_line":3,
  "fontsize":64,
  "background_music_url": "https://example.com/music.mp3",
  "music_volume": 0.3,
  "fade_music": true
}

How to run with caption:
1. Start the server:
   ```
   uvicorn clip_server:app --host 0.0.0.0 --port 8000
   ```
2. Make a POST request to `/clip_with_caption` (e.g. using `curl`):
   ```
   curl -X POST "http://localhost:8000/clip_with_caption" ^
     -H "Content-Type: application/json" ^
     -d "{\"url\": \"https://www.youtube.com/watch?v=v52S3LBFZJs\", \"start\": 10, \"end\": 40, \"burn\": true, \"resolution\": \"1080x1920\", \"background_music_url\": \"https://example.com/music.mp3\", \"music_volume\": 0.3, \"fade_music\": true}" ^
     --output output.mp4
   ```
   Or use the interactive docs at http://localhost:8000/docs

Requirements: same as before (ffmpeg, yt-dlp, optional ASR backends)
"""

import os
import shlex
import math
import subprocess
import tempfile
import asyncio
import logging
import importlib
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, HttpUrl
from fastapi.responses import FileResponse
import requests

# load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clip_server")

# Optional OpenAI API key for remote Whisper
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BG_IMAGE_PATH = os.path.abspath("assets/bg_default.jpg")

# Create default background if it doesn't exist
def ensure_default_background():
    os.makedirs("assets", exist_ok=True)
    if not os.path.exists(BG_IMAGE_PATH):
        # Create a simple black background using ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s=1920x1080:d=1",
            "-frames:v", "1",
            BG_IMAGE_PATH
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=10)
            logger.info("Created default black background at %s", BG_IMAGE_PATH)
        except Exception as e:
            logger.warning("Could not create default background: %s", e)

ensure_default_background()

app = FastAPI(title="Clip + Caption Server (OpenAI-first, fallback local) - resizable")

# ----------------- Request models -----------------
class ClipRequest(BaseModel):
    url: HttpUrl
    start: float
    end: float
    resolution: Optional[str] = None  # e.g. "1920x1080" or "1080x1920"
    background_image_url: Optional[str] = None  # ⭐ URL background image dari CDN
    background_music_url: Optional[str] = None  # ⭐ URL musik dari storage/CDN
    music_volume: Optional[float] = 0.3  # Volume musik (0.0-1.0), 0.3 = 30%
    fade_music: Optional[bool] = True  # Fade in/out musik

class ClipCaptionRequest(ClipRequest):
    model: Optional[str] = "small"   # local model size (tiny,base,small,medium,large) or "whisper-1" to prefer OpenAI
    burn: Optional[bool] = False
    language: Optional[str] = None
    words_per_line: Optional[int] = 3
    fontsize: Optional[int] = 56
    margin_v: Optional[int] = 50
    # ⭐ TAMBAHAN UNTUK BACKGROUND MUSIC
    background_music_url: Optional[str] = None  # URL musik dari storage/CDN
    music_volume: Optional[float] = 0.3  # Volume musik (0.0-1.0), 0.3 = 30%
    fade_music: Optional[bool] = True  # Fade in/out musik

# ----------------- Subprocess runner -----------------

def run_cmd(cmd: List[str], timeout: int = None) -> subprocess.CompletedProcess:
    logger.debug("Run command: %s", " ".join(shlex.quote(c) for c in cmd))
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return cp

# ----------------- Helpers for resolution parsing and resizing -----------------

def parse_resolution(res: Optional[str]) -> Optional[Tuple[int, int]]:
    if not res:
        return None
    if isinstance(res, str) and "x" in res:
        try:
            w_s, h_s = res.lower().split("x")
            w = int(w_s)
            h = int(h_s)
            if w <= 0 or h <= 0:
                raise ValueError("width/height must be > 0")
            return (w, h)
        except Exception as e:
            raise ValueError(f"Invalid resolution format '{res}', expected WIDTHxHEIGHT")
    raise ValueError(f"Invalid resolution format '{res}', expected WIDTHxHEIGHT")


def resize_video(input_path: str, out_path: str, resolution: Tuple[int, int], mode: str = "fill"):
    """
    Resize video to exact resolution.
    mode:
      - "fit": scale preserving aspect, then pad (no cropping) -> may produce black bars
      - "fill": scale preserving aspect then center-crop to fill (zoom & crop)
    """
    w, h = resolution

    if mode == "fit":
        # scale down/up to fit and pad to exact size (kept from earlier)
        vf = (
            f"scale=iw*min({w}/iw\\,{h}/ih):ih*min({w}/iw\\,{h}/ih),"
            f"pad={w}:{h}:({w}-iw*min({w}/iw\\,{h}/ih))/2:({h}-ih*min({w}/iw\\,{h}/ih))/2"
        )
    else:
        # fill (zoom & center-crop)
        vf = (
            f"scale=iw*max({w}/iw\\,{h}/ih):ih*max({w}/iw\\,{h}/ih),"
            f"crop={w}:{h}"
        )

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        out_path
    ]
    cp = run_cmd(cmd, timeout=600)
    if cp.returncode != 0:
        logger.error("resize ffmpeg failed: %s", cp.stderr or cp.stdout)
        raise RuntimeError("resize ffmpeg failed: " + (cp.stderr or cp.stdout or ""))
    logger.info("Resized video to %dx%d (mode=%s) -> %s", w, h, mode, out_path)

def resize_video_with_bg(
    input_video: str,
    out_path: str,
    resolution: Tuple[int, int],
    bg_image: str
):
    """
    Resize video FIT (no crop) and fill empty area with background image.
    """
    # Check if background exists, otherwise fallback to black padding
    if not os.path.exists(bg_image):
        logger.warning(f"Background image not found: {bg_image}, using black padding instead")
        w, h = resolution
        vf = (
            f"scale=iw*min({w}/iw\\,{h}/ih):ih*min({w}/iw\\,{h}/ih),"
            f"pad={w}:{h}:({w}-iw*min({w}/iw\\,{h}/ih))/2:({h}-ih*min({w}/iw\\,{h}/ih))/2:black"
        )
        cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            out_path
        ]
        cp = run_cmd(cmd, timeout=600)
        if cp.returncode != 0:
            logger.error("resize with black padding failed: %s", cp.stderr or cp.stdout)
            raise RuntimeError("resize with black padding failed")
        logger.info("Resized video with black padding to %dx%d -> %s", w, h, out_path)
        return

    w, h = resolution
    bg = bg_image.replace("\\", "/")
    inp = input_video.replace("\\", "/")

    # Get video duration first
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        inp
    ]
    probe_result = run_cmd(probe_cmd, timeout=30)
    try:
        duration = float(probe_result.stdout.strip())
    except:
        duration = 30.0  # fallback

    # Simplified filter_complex - remove fade effect if causing issues
    filter_complex = (
        f"[0:v]scale=w=iw*min({w}/iw\\,{h}/ih):h=ih*min({w}/iw\\,{h}/ih),setsar=1[vid];"
        f"[1:v]scale={w}:{h},setsar=1,loop=loop=-1:size=1:start=0[bg];"
        f"[bg][vid]overlay=(W-w)/2:(H-h)/2:shortest=1[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", inp,
        "-loop", "1", "-t", str(duration), "-i", bg,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        "-shortest",
        out_path
    ]

    logger.info("Running ffmpeg with background image: %s", " ".join(cmd))
    cp = run_cmd(cmd, timeout=600)
    
    if cp.returncode != 0:
        logger.error("resize with bg failed - stdout: %s", cp.stdout)
        logger.error("resize with bg failed - stderr: %s", cp.stderr)
        
        # Fallback to black padding if background fails
        logger.warning("Falling back to black padding resize")
        vf = (
            f"scale=iw*min({w}/iw\\,{h}/ih):ih*min({w}/iw\\,{h}/ih),"
            f"pad={w}:{h}:({w}-iw*min({w}/iw\\,{h}/ih))/2:({h}-ih*min({w}/iw\\,{h}/ih))/2:black"
        )
        cmd_fallback = [
            "ffmpeg", "-y", "-i", inp,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            out_path
        ]
        cp_fallback = run_cmd(cmd_fallback, timeout=600)
        if cp_fallback.returncode != 0:
            logger.error("fallback resize also failed: %s", cp_fallback.stderr or cp_fallback.stdout)
            raise RuntimeError("resize with bg and fallback both failed: " + (cp.stderr or cp.stdout or ""))
        logger.info("Fallback resize successful")
        return

    logger.info("Resized video with background to %dx%d -> %s", w, h, out_path)


def download_video_with_ytdlp(youtube_url: str, out_path: str):
    """
    Robust yt-dlp downloader to avoid 403 / nsig issues.
    """
    out_template = out_path.replace("\\", "/")

    cmd = [
        "yt-dlp",
        youtube_url,

        # format
        "-f", "bv*+ba/b",

        # output
        "-o", out_template,
        "--merge-output-format", "mp4",
        "--no-part",

        # IMPORTANT anti-403 flags
        "--extractor-args", "youtube:player_client=android",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "--referer", "https://www.youtube.com/",
        "--no-check-certificate",
        "--geo-bypass",

        # retries
        "--retries", "5",
        "--fragment-retries", "5",
        "--retry-sleep", "1",
    ]

    cp = run_cmd(cmd, timeout=900)

    if cp.returncode != 0:
        logger.error("yt-dlp download failed: %s", cp.stderr or cp.stdout)
        raise RuntimeError("yt-dlp download failed: " + (cp.stderr or cp.stdout or ""))

    if not os.path.exists(out_path):
        raise RuntimeError(f"yt-dlp failed: output file not found ({out_path})")

    logger.info("Downloaded video to %s", out_path)


# ----------------- FFmpeg cutting helpers -----------------
def run_ffmpeg_cut_local(input_path: str, start_sec: float, duration_sec: float, out_path: str):
    # fail early if input is not a local file (avoid ffmpeg trying remote URL -> 403)
    if not os.path.exists(input_path):
        if isinstance(input_path, str) and (input_path.startswith("http://") or input_path.startswith("https://")):
            raise RuntimeError(f"ffmpeg single input failed: input is a remote URL (access denied), expected local file: {input_path}")
        raise RuntimeError(f"ffmpeg single input failed: input file not found: {input_path}")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec), "-t", str(duration_sec),
        "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
        out_path
    ]
    cp = run_cmd(cmd, timeout=600)
    if cp.returncode != 0:
        logger.error("ffmpeg cut failed: %s", cp.stderr or cp.stdout)
        raise RuntimeError("ffmpeg cut failed: " + (cp.stderr or cp.stdout or ""))
    logger.info("Cut video to %s", out_path)


# ----------------- audio extraction -----------------
def extract_audio_wav(input_video: str, out_wav: str):
    cmd = ["ffmpeg", "-i", input_video, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", out_wav]
    cp = run_cmd(cmd, timeout=120)
    if cp.returncode != 0:
        logger.error("extract audio failed: %s", cp.stderr or cp.stdout)
        raise RuntimeError("extract audio failed: " + (cp.stderr or cp.stdout or ""))
    logger.info("extracted audio: %s", out_wav)

# ----------------- Audio mixing helpers -----------------
def download_music_file(music_url: str, out_path: str):
    """
    Download music file from URL.
    """
    try:
        logger.info("Downloading music from: %s", music_url)
        resp = requests.get(music_url, stream=True, timeout=60)
        resp.raise_for_status()
        
        with open(out_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("Downloaded music file to %s", out_path)
    except Exception as e:
        logger.error("Failed to download music: %s", e)
        raise RuntimeError(f"Failed to download music: {e}")


def download_image_file(image_url: str, out_path: str):
    """
    Download image file from URL (for background).
    """
    try:
        logger.info("Downloading background image from: %s", image_url)
        resp = requests.get(image_url, stream=True, timeout=60)
        resp.raise_for_status()
        
        with open(out_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("Downloaded background image to %s", out_path)
    except Exception as e:
        logger.error("Failed to download background image: %s", e)
        raise RuntimeError(f"Failed to download background image: {e}")


def add_background_music(
    input_video: str,
    music_path: str,
    out_path: str,
    music_volume: float = 0.3,
    fade_duration: float = 2.0
):
    """
    Add background music to video with volume control and fade in/out.
    """
    if not os.path.exists(music_path):
        raise RuntimeError(f"Music file not found: {music_path}")
    
    music_path_norm = music_path.replace("\\", "/")
    input_video_norm = input_video.replace("\\", "/")
    
    # Get video duration
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_video_norm
    ]
    probe_result = run_cmd(probe_cmd, timeout=30)
    try:
        video_duration = float(probe_result.stdout.strip())
    except:
        video_duration = 30.0
    
    logger.info("Video duration: %.2f seconds, adding music with volume: %.2f", video_duration, music_volume)
    
    # Audio filter: loop music, adjust volume, fade in/out, mix with original audio
    if fade_duration > 0:
        audio_filter = (
            f"[1:a]aloop=loop=-1:size=2e+09,"
            f"volume={music_volume},"
            f"afade=t=in:st=0:d={fade_duration},"
            f"afade=t=out:st={max(0, video_duration-fade_duration)}:d={fade_duration},"
            f"atrim=0:{video_duration}[music];"
            f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
    else:
        audio_filter = (
            f"[1:a]aloop=loop=-1:size=2e+09,"
            f"volume={music_volume},"
            f"atrim=0:{video_duration}[music];"
            f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video_norm,
        "-stream_loop", "-1",
        "-i", music_path_norm,
        "-filter_complex", audio_filter,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_path
    ]
    
    logger.info("Running ffmpeg to add background music...")
    cp = run_cmd(cmd, timeout=600)
    if cp.returncode != 0:
        logger.error("add background music failed: %s", cp.stderr or cp.stdout)
        raise RuntimeError("add background music failed: " + (cp.stderr or cp.stdout or ""))
    
    logger.info("Added background music to video -> %s", out_path)


# ----------------- Local ASR helpers -----------------
_HAS_FASTER = importlib.util.find_spec("faster_whisper") is not None
_HAS_WHISPER = importlib.util.find_spec("whisper") is not None

def transcribe_audio_local(audio_path: str, model_size: str = "small", language: Optional[str] = None) -> List[dict]:
    if _HAS_FASTER:
        from faster_whisper import WhisperModel
        logger.info("Transcribing with faster_whisper model=%s", model_size)
        model = WhisperModel(model_size, device="auto", compute_type="float16")
        segments = []
        for seg in model.transcribe(audio_path, language=language, beam_size=5):
            segments.append({"start": float(seg.start), "end": float(seg.end), "text": str(seg.text).strip()})
        return segments
    elif _HAS_WHISPER:
        import whisper
        logger.info("Transcribing with openai-whisper model=%s", model_size)
        model = whisper.load_model(model_size)
        kwargs = {}
        if language:
            kwargs["language"] = language
        res = model.transcribe(audio_path, **kwargs)
        segments = []
        for s in res.get("segments", []):
            segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": str(s["text"]).strip()})
        return segments
    else:
        raise RuntimeError("No local ASR backend installed (install faster-whisper or openai-whisper)")

# ----------------- OpenAI API ASR (fallback) -----------------

def transcribe_audio_openai(audio_path: str, model: str = "whisper-1", language: Optional[str] = None) -> List[dict]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"model": model, "response_format": "verbose_json"}
    if language:
        data["language"] = language
    with open(audio_path, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=120)
    if resp.status_code != 200:
        logger.error("OpenAI ASR error: %s", resp.text)
        raise RuntimeError(f"OpenAI ASR failed: {resp.status_code} {resp.text}")
    result = resp.json()
    segments = []
    if "segments" in result:
        for seg in result["segments"]:
            segments.append({"start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)), "text": seg.get("text", "").strip()})
    else:
        segments.append({"start": 0.0, "end": 1.0, "text": result.get("text", "").strip()})
    return segments

# ----------------- Prefer OpenAI then local wrapper -----------------
def transcribe_audio_prefer_openai(audio_path: str, openai_model: str = "whisper-1", local_model_size: str = "small", language: Optional[str] = None) -> List[dict]:
    # Try OpenAI first if key present
    if OPENAI_API_KEY:
        try:
            logger.info("Trying OpenAI ASR first")
            segments = transcribe_audio_openai(audio_path, model=openai_model, language=language)
            if segments and isinstance(segments, list) and len(segments) > 0:
                logger.info("OpenAI ASR success")
                return segments
            logger.warning("OpenAI ASR returned empty segments; falling back to local")
        except Exception as e:
            logger.warning("OpenAI ASR failed or quota issue: %s. Falling back to local ASR.", str(e))

    # Fallback local
    try:
        logger.info("Trying local ASR model=%s", local_model_size)
        segments = transcribe_audio_local(audio_path, model_size=local_model_size, language=language)
        if segments and isinstance(segments, list) and len(segments) > 0:
            logger.info("Local ASR success")
            return segments
        logger.warning("Local ASR returned no segments")
    except Exception as le:
        logger.exception("Local ASR failed: %s", le)

    raise RuntimeError("Transcription failed: both OpenAI API and local ASR unavailable or failed")

# ----------------- ASS generation (karaoke-style) -----------------
import html

def seconds_to_ass_time(s: float) -> str:
    if s < 0: s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    cs = int(round((s - math.floor(s)) * 100))
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"

def generate_ass_from_segments(
    segments: List[dict],
    ass_out_path: str,
    words_per_line: int = 3,
    fontname: str = "Arial Black",
    fontsize: int = 56,
    margin_v: int = 50,
    outline: int = 3,
    shadow: int = 1,
    primary_color: str = "&H00FFFFFF"  # ⭐ UBAH INI untuk warna text
):
    def safe_text(t: str) -> str:
        t = t.replace("-->", "->")
        t = t.replace("{", "\\{").replace("}", "\\}")
        return html.escape(t)

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "Timer: 100.0000",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # ⭐ UBAH BARIS INI:
        # BackColour: &H00000000 = hitam transparan (00 di awal = full transparent)
        # OutlineColour: &H00000000 = outline hitam
        # PrimaryColour: primary_color = warna text utama
        f"Style: MyStyle,{fontname},{fontsize},{primary_color},&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,10,10,{margin_v},1",
        # Text putih, outline hitam, no background (current)
        # f"Style: MyStyle,{fontname},{fontsize},&H00FFFFFF,&H00000000,&H00000000,&H00000000,..."

        # Text kuning, outline hitam, no background
        # f"Style: MyStyle,{fontname},{fontsize},&H0000FFFF,&H00000000,&H00000000,&H00000000,..."

        # Text putih, outline merah, no background
        # f"Style: MyStyle,{fontname},{fontsize},&H00FFFFFF,&H00000000,&H000000FF,&H00000000,..."

        # Text putih, outline hitam, background merah semi-transparan
        # f"Style: MyStyle,{fontname},{fontsize},&H00FFFFFF,&H00000000,&H00000000,&H80000080,..."
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    dialogues: List[str] = []

    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start + 0.5))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        words = text.split()
        total_dur = max(0.1, seg_end - seg_start)
        per_word = total_dur / max(1, len(words))

        for i in range(0, len(words), words_per_line):
            chunk_words = words[i:i + words_per_line]
            chunk_text = " ".join(chunk_words)

            chunk_start = seg_start + i * per_word
            chunk_end = min(seg_end, chunk_start + per_word * len(chunk_words))

            start_ts = seconds_to_ass_time(chunk_start)
            end_ts = seconds_to_ass_time(chunk_end)

            dialogue = (
                f"Dialogue: 0,{start_ts},{end_ts},MyStyle,,"
                f"0,0,0,,{safe_text(chunk_text)}"
            )
            dialogues.append(dialogue)

    with open(ass_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + dialogues))

    logger.info("Wrote ASS file: %s (%d dialogues)", ass_out_path, len(dialogues))

# ----------------- subtitle embed/burn robust -----------------
def embed_subtitles_soft(input_video: str, srt_or_ass: str, out_path: str):
    norm = srt_or_ass.replace("\\", "/")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", norm,
        "-map", "0:v",
        "-map", "0:a?",
        "-map", "1:0",
        # re-encode video to ensure compatibility with boxed resolution if needed
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        out_path
    ]
    cp = run_cmd(cmd, timeout=180)
    if cp.returncode == 0:
        logger.info("embed subtitles success: %s", out_path)
        return
    logger.warning("embed subtitles failed, trying MKV fallback: %s", cp.stderr or cp.stdout)
    out_mkv = out_path.rsplit(".", 1)[0] + ".mkv"
    cmd2 = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", norm,
        "-map", "0",
        "-map", "1:0",
        "-c", "copy",
        "-c:s", "srt",
        out_mkv
    ]
    cp2 = run_cmd(cmd2, timeout=120)
    if cp2.returncode == 0:
        os.replace(out_mkv, out_path)
        logger.info("embed fallback MKV success: %s", out_path)
        return
    logger.error("embed fallback failed: %s", cp2.stderr or cp2.stdout)
    raise RuntimeError("embed subtitles failed: " + (cp.stderr or cp.stdout or ""))


def burn_subtitles_hard(input_video: str, ass_file: str, out_path: str):
    ass_path = None
    try:
        ass_norm = ass_file.replace("\\", "/")
        # if user passed .srt, convert to ass
        if ass_norm.lower().endswith(".srt"):
            fd, tmp_ass = tempfile.mkstemp(suffix=".ass")
            os.close(fd)
            cp_conv = run_cmd(["ffmpeg", "-y", "-f", "srt", "-i", ass_norm, tmp_ass], timeout=40)
            if cp_conv.returncode != 0:
                logger.error("SRT->ASS conversion failed: %s", cp_conv.stderr or cp_conv.stdout)
                raise RuntimeError("SRT->ASS conversion failed")
            ass_norm = tmp_ass
            ass_path = tmp_ass
        # escape ":" and spaces and single quotes
        ass_esc = ass_norm.replace(":", r"\:").replace(" ", r"\ ").replace("'", r"\'")
        vf = f"ass='{ass_esc}'"
        cmd = ["ffmpeg", "-y", "-i", input_video, "-vf", vf, "-c:a", "copy", out_path]
        cp = run_cmd(cmd, timeout=300)
        if cp.returncode == 0:
            logger.info("burn ASS success: %s", out_path)
            return
        logger.error("burn ASS failed: %s", cp.stderr or cp.stdout)
        raise RuntimeError("burn ASS failed: " + (cp.stderr or cp.stdout or ""))
    finally:
        if ass_path and os.path.exists(ass_path):
            try: os.remove(ass_path)
            except Exception: pass

# ----------------- cleanup -----------------
def remove_file_later(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info("removed temp file %s", path)
    except Exception as e:
        logger.warning("failed to remove temp file %s: %s", path, e)

# ----------------- API endpoints -----------------
@app.post("/clip")
async def clip_endpoint(req: ClipRequest, background: BackgroundTasks):
    start = float(req.start); end = float(req.end)
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be greater than start")
    duration = end - start

    tmp_download = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_download_path = tmp_download.name; tmp_download.close()
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_out_path = tmp_out.name; tmp_out.close()
    tmp_resized = None
    tmp_bg_image = None
    tmp_music_path = None
    tmp_with_music = None

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, download_video_with_ytdlp, str(req.url), tmp_download_path)
        await loop.run_in_executor(None, run_ffmpeg_cut_local, tmp_download_path, start, duration, tmp_out_path)
    except Exception as e:
        for p in (tmp_download_path, tmp_out_path):
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        logger.exception("yt-dlp or ffmpeg cut failed")
        raise HTTPException(status_code=500, detail=f"yt-dlp or ffmpeg failed: {e}")

    # optional resize
    try:
        res = parse_resolution(req.resolution)
        if res:
            # ⭐ Download background image if URL provided
            bg_path = BG_IMAGE_PATH
            if req.background_image_url:
                img_ext = req.background_image_url.split('?')[0].split('.')[-1].lower()
                if img_ext not in ['jpg', 'jpeg', 'png', 'webp', 'bmp']:
                    img_ext = 'jpg'
                fd_bg, tmp_bg_image = tempfile.mkstemp(suffix=f".{img_ext}")
                os.close(fd_bg)
                await loop.run_in_executor(None, download_image_file, req.background_image_url, tmp_bg_image)
                bg_path = tmp_bg_image
            
            fd, tmp_resized = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            await loop.run_in_executor(None, resize_video_with_bg, tmp_out_path, tmp_resized, res, bg_path)
            final_output = tmp_resized
            background.add_task(remove_file_later, tmp_out_path)
            if tmp_bg_image:
                background.add_task(remove_file_later, tmp_bg_image)
        else:
            final_output = tmp_out_path
    except ValueError as ve:
        for p in (tmp_download_path, tmp_out_path, tmp_resized, tmp_bg_image):
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        for p in (tmp_download_path, tmp_out_path, tmp_resized, tmp_bg_image):
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        logger.exception("resize failed")
        raise HTTPException(status_code=500, detail=f"resize failed: {e}")

    # ⭐ ADD BACKGROUND MUSIC (OPTIONAL)
    final_video_path = final_output
    if req.background_music_url:
        try:
            logger.info("Processing background music from URL: %s", req.background_music_url)
            
            # Download music file
            music_ext = req.background_music_url.split('?')[0].split('.')[-1].lower()
            if music_ext not in ['mp3', 'wav', 'aac', 'm4a', 'ogg', 'flac']:
                music_ext = 'mp3'
            
            fd_music, tmp_music_path = tempfile.mkstemp(suffix=f".{music_ext}")
            os.close(fd_music)
            
            logger.info("Downloading music to: %s", tmp_music_path)
            await loop.run_in_executor(None, download_music_file, req.background_music_url, tmp_music_path)
            
            # Add music to video
            fd_with_music, tmp_with_music = tempfile.mkstemp(suffix=".mp4")
            os.close(fd_with_music)
            
            fade_duration = 2.0 if req.fade_music else 0.0
            music_volume = float(req.music_volume) if req.music_volume is not None else 0.3
            
            logger.info("Adding music with volume=%.2f, fade=%.1fs", music_volume, fade_duration)
            await loop.run_in_executor(
                None,
                add_background_music,
                final_output,
                tmp_music_path,
                tmp_with_music,
                music_volume,
                fade_duration
            )
            
            # Update final output to the one with music
            final_video_path = tmp_with_music
            logger.info("Background music added successfully to: %s", final_video_path)
            
        except Exception as e:
            logger.exception("Failed to add background music: %s", e)
            # Continue without music if failed, cleanup temp files
            final_video_path = final_output
            if tmp_music_path and os.path.exists(tmp_music_path):
                try: 
                    os.remove(tmp_music_path)
                    tmp_music_path = None
                except: pass
            if tmp_with_music and os.path.exists(tmp_with_music):
                try: 
                    os.remove(tmp_with_music)
                    tmp_with_music = None
                except: pass

    # Cleanup temp files
    background.add_task(remove_file_later, tmp_download_path)
    if tmp_bg_image and os.path.exists(tmp_bg_image):
        background.add_task(remove_file_later, tmp_bg_image)
    if tmp_music_path and os.path.exists(tmp_music_path):
        background.add_task(remove_file_later, tmp_music_path)
    
    # If we created a video with music, clean the one without music
    if tmp_with_music and final_video_path == tmp_with_music and final_output != final_video_path:
        background.add_task(remove_file_later, final_output)
    
    # Clean final output after response is sent
    background.add_task(remove_file_later, final_video_path)
    
    filename = f"clip-{Path(final_video_path).stem}.mp4"
    return FileResponse(final_video_path, media_type="video/mp4", filename=filename)

@app.post("/clip_with_caption")
async def clip_with_caption_endpoint(req: ClipCaptionRequest, background: BackgroundTasks):
    start = float(req.start); end = float(req.end)
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be greater than start")
    duration = end - start

    tmp_download = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp_download_path = tmp_download.name; tmp_download.close()
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp_video_path = tmp_video.name; tmp_video.close()
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav"); tmp_wav_path = tmp_wav.name; tmp_wav.close()
    tmp_ass = tempfile.NamedTemporaryFile(delete=False, suffix=".ass"); tmp_ass_path = tmp_ass.name; tmp_ass.close()
    out_final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); out_final_path = out_final.name; out_final.close()
    tmp_video_resized = None
    tmp_music_path = None
    tmp_with_music = None
    tmp_bg_image = None

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, download_video_with_ytdlp, str(req.url), tmp_download_path)
        await loop.run_in_executor(None, run_ffmpeg_cut_local, tmp_download_path, start, duration, tmp_video_path)
    except Exception as e:
        for p in (tmp_download_path, tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path):
            try: os.remove(p)
            except Exception: pass
        logger.exception("yt-dlp or ffmpeg cut failed")
        raise HTTPException(status_code=500, detail=f"yt-dlp or ffmpeg cut failed: {e}")

    # 2) extract audio
    try:
        await loop.run_in_executor(None, extract_audio_wav, tmp_video_path, tmp_wav_path)
    except Exception as e:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path):
            try: os.remove(p)
            except Exception: pass
        logger.exception("extract audio failed")
        raise HTTPException(status_code=500, detail=f"extract audio failed: {e}")

    # 3) transcribe (prefer OpenAI API then local)
    try:
        local_model_size = req.model if req.model in ("tiny","base","small","medium","large") else "small"
        segments = await loop.run_in_executor(None, transcribe_audio_prefer_openai, tmp_wav_path, "whisper-1", local_model_size, req.language)
        if not segments:
            logger.warning("transcription returned no segments")
    except Exception as e:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path):
            try: os.remove(p)
            except Exception: pass
        logger.exception("transcription failed")
        raise HTTPException(status_code=500, detail=f"transcription failed: {e}")

    # 4) generate ASS (karaoke style)
    try:
        words_per_line = int(req.words_per_line) if req.words_per_line else 3
        fontsize = int(req.fontsize) if req.fontsize else 56
        margin_v = int(req.margin_v) if req.margin_v else 50
        await loop.run_in_executor(None, generate_ass_from_segments, segments, tmp_ass_path, words_per_line, "Arial", fontsize, margin_v)
    except Exception as e:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path):
            try: os.remove(p)
            except Exception: pass
        logger.exception("generate ASS failed")
        raise HTTPException(status_code=500, detail=f"generate ASS failed: {e}")

    # optional resize before burn/embed so final matches requested resolution
    try:
        res = parse_resolution(req.resolution)
        if res:
            # ⭐ Download background image if URL provided
            bg_path = BG_IMAGE_PATH
            if req.background_image_url:
                img_ext = req.background_image_url.split('?')[0].split('.')[-1].lower()
                if img_ext not in ['jpg', 'jpeg', 'png', 'webp', 'bmp']:
                    img_ext = 'jpg'
                fd_bg, tmp_bg_image = tempfile.mkstemp(suffix=f".{img_ext}")
                os.close(fd_bg)
                await loop.run_in_executor(None, download_image_file, req.background_image_url, tmp_bg_image)
                bg_path = tmp_bg_image
            
            fd, tmp_video_resized = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            await loop.run_in_executor(None, resize_video_with_bg, tmp_video_path, tmp_video_resized, res, bg_path)
            # keep resized path as input for subtitle steps
            working_video_for_sub = tmp_video_resized
        else:
            working_video_for_sub = tmp_video_path
    except ValueError as ve:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path, tmp_video_resized if tmp_video_resized else "", tmp_bg_image):
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path, tmp_video_resized if tmp_video_resized else "", tmp_bg_image):
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        logger.exception("resize failed")
        raise HTTPException(status_code=500, detail=f"resize failed: {e}")

    # 5) burn or embed onto working_video_for_sub
    try:
        if req.burn:
            await loop.run_in_executor(None, burn_subtitles_hard, working_video_for_sub, tmp_ass_path, out_final_path)
        else:
            await loop.run_in_executor(None, embed_subtitles_soft, working_video_for_sub, tmp_ass_path, out_final_path)
    except Exception as e:
        for p in (tmp_video_path, tmp_wav_path, tmp_ass_path, out_final_path, tmp_video_resized if tmp_video_resized else ""):
            try: os.remove(p)
            except Exception: pass
        logger.exception("subtitle processing failed")
        raise HTTPException(status_code=500, detail=f"subtitle processing failed: {e}")

    # ⭐ 6) TAMBAHKAN BACKGROUND MUSIC (OPTIONAL)
    final_output_path = out_final_path
    if req.background_music_url:
        try:
            logger.info("Processing background music from URL: %s", req.background_music_url)
            
            # Download music file
            music_ext = req.background_music_url.split('?')[0].split('.')[-1].lower()
            if music_ext not in ['mp3', 'wav', 'aac', 'm4a', 'ogg', 'flac']:
                music_ext = 'mp3'
            
            fd_music, tmp_music_path = tempfile.mkstemp(suffix=f".{music_ext}")
            os.close(fd_music)
            
            logger.info("Downloading music to: %s", tmp_music_path)
            await loop.run_in_executor(None, download_music_file, req.background_music_url, tmp_music_path)
            
            # Add music to video
            fd_with_music, tmp_with_music = tempfile.mkstemp(suffix=".mp4")
            os.close(fd_with_music)
            
            fade_duration = 2.0 if req.fade_music else 0.0
            music_volume = float(req.music_volume) if req.music_volume is not None else 0.3
            
            logger.info("Adding music with volume=%.2f, fade=%.1fs", music_volume, fade_duration)
            await loop.run_in_executor(
                None,
                add_background_music,
                out_final_path,
                tmp_music_path,
                tmp_with_music,
                music_volume,
                fade_duration
            )
            
            # Update final output to the one with music
            final_output_path = tmp_with_music
            logger.info("Background music added successfully to: %s", final_output_path)
            
        except Exception as e:
            logger.exception("Failed to add background music: %s", e)
            # Continue tanpa musik jika gagal, cleanup temp files
            final_output_path = out_final_path
            if tmp_music_path and os.path.exists(tmp_music_path):
                try: 
                    os.remove(tmp_music_path)
                    tmp_music_path = None
                except: pass
            if tmp_with_music and os.path.exists(tmp_with_music):
                try: 
                    os.remove(tmp_with_music)
                    tmp_with_music = None
                except: pass

    # schedule cleanup - clean all temp files EXCEPT final output
    for p in (tmp_download_path, tmp_video_path, tmp_wav_path, tmp_ass_path, tmp_video_resized, tmp_music_path, tmp_bg_image):
        if p and os.path.exists(p):
            background.add_task(remove_file_later, p)
    
    # If we created a video with music, clean the one without music
    if tmp_with_music and final_output_path == tmp_with_music and out_final_path != final_output_path:
        background.add_task(remove_file_later, out_final_path)
    
    # Clean final output after response is sent
    background.add_task(remove_file_later, final_output_path)

    filename = f"clip-caption-{Path(final_output_path).stem}.mp4"
    return FileResponse(final_output_path, media_type="video/mp4", filename=filename)

@app.get("/health")
async def health():
    return {"ok": True}
