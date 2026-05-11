import json
import os

from dotenv import load_dotenv
load_dotenv()
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from app.services.pose_extract import (
    extract_pose_to_dict,
    detect_swing_phases,
    compute_golf_metrics,
    generate_coaching_feedback,
    build_swing_report,
    save_keyframe_images,
    save_annotated_video,
)
from app.services.ai_coach import generate_ai_coaching, stream_chat
from app.services.reference_matcher import (
    load_reference_db, save_reference_db, add_swing_to_db,
    extract_phase_vectors, score_against_reference,
)


app = FastAPI(
    title="SwingAI Backend",
    description="Backend API for SwingAI golf swing analysis",
    version="0.3.0"
)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated artifacts (images/videos) so the frontend can display them
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../SwingAI
OUTPUTS_DIR = PROJECT_ROOT / "backend" / "outputs"
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.on_event("startup")
def clear_outputs_on_startup() -> None:
    visuals_dir = OUTPUTS_DIR / "visuals"
    if visuals_dir.exists():
        for f in visuals_dir.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass


class ChatHistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    swing_context: Dict[str, Any] = {}
    history: List[ChatHistoryItem] = []


@app.post("/chat")
async def chat(req: ChatRequest):
    """Stream a Claude response to a question about the golfer's swing."""
    async def stream():
        try:
            async for chunk in stream_chat(
                message=req.message,
                swing_context=req.swing_context,
                history=[h.model_dump() for h in req.history],
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'text': f'Sorry, something went wrong: {e}'})}\n\n"
        finally:
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def read_root():
    return {
        "message": "Welcome to SwingAI Backend",
        "service": "golf swing analysis",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "ok": True
    }


@app.post("/analyze/phases")
async def analyze_phases(
    video: UploadFile = File(...),
    ball_x: Optional[float] = Form(None),
    ball_y: Optional[float] = Form(None),
    view: str = "side_on"
):
    """
    Upload a golf swing video and return detected swing phases:
    - address
    - top of backswing
    - impact
    - tempo metrics
    """
    # Save uploaded video to a temporary file
    suffix = os.path.splitext(video.filename or "upload.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await video.read()
        tmp.write(content)

    try:
        # Run pose extraction + phase detection
        pose = extract_pose_to_dict(tmp_path, view=view)
        ball_xy = (float(ball_x), float(ball_y)) if (ball_x is not None and ball_y is not None) else None
        phases = detect_swing_phases(pose, ball_xy=ball_xy)

        return JSONResponse(
            {
                "pose_meta": pose.get("meta", {}),
                "phases": phases,
                "ball_xy": list(ball_xy) if ball_xy is not None else None,
            }
        )
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.post("/analyze/report")
async def analyze_report(
    video: UploadFile = File(...),
    ball_x: Optional[float] = Form(None),
    ball_y: Optional[float] = Form(None),
    view: str = "side_on",
    handedness: str = "right",
):
    """
    Upload a golf swing video and return a full report:
    - phases (address/top/impact + tempo)
    - computed metrics
    - coaching feedback
    """
    suffix = os.path.splitext(video.filename or "upload.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await video.read()
        tmp.write(content)

    try:
        pose = extract_pose_to_dict(tmp_path, view=view)
        ball_xy = (float(ball_x), float(ball_y)) if (ball_x is not None and ball_y is not None) else None
        phases = detect_swing_phases(pose, ball_xy=ball_xy)
        metrics = compute_golf_metrics(pose, phases, handedness=handedness)
        feedback = generate_ai_coaching(metrics, handedness=handedness) or generate_coaching_feedback(metrics)
        report = {"phases": phases, "metrics": metrics, "feedback": feedback}

        db = load_reference_db()
        ref_score = score_against_reference(extract_phase_vectors(pose, phases, handedness), db)

        return JSONResponse(
            {
                "pose_meta": pose.get("meta", {}),
                "report": report,
                "ball_xy": list(ball_xy) if ball_xy is not None else None,
                "reference_score": ref_score,
            }
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.post("/analyze/visuals")
async def analyze_visuals(
    video: UploadFile = File(...),
    ball_x: Optional[float] = Form(None),
    ball_y: Optional[float] = Form(None),
    view: str = "side_on",
    handedness: str = "right",
):
    """
    Upload a golf swing video and save 3 keyframe images with pose overlay:
    - ADDRESS
    - TOP
    - IMPACT

    Returns file paths to the saved images (v1).
    """
    suffix = os.path.splitext(video.filename or "upload.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await video.read()
        tmp.write(content)

    try:
        pose = extract_pose_to_dict(tmp_path, view=view)
        ball_xy = (float(ball_x), float(ball_y)) if (ball_x is not None and ball_y is not None) else None
        phases = detect_swing_phases(pose, ball_xy=ball_xy)

        # Save images under backend/outputs/visuals
        out_dir = OUTPUTS_DIR / "visuals"

        # Prefix based on original filename (safe-ish) or a fallback
        base = os.path.splitext(video.filename or "swing")[0]
        base = "".join([c for c in base if c.isalnum() or c in ("-", "_")])[:40] or "swing"
        prefix = f"{base}_{phases['address']['frame']}_{phases['impact']['frame']}"

        paths = save_keyframe_images(pose, phases, out_dir=out_dir, prefix=prefix)

        # Convert filesystem paths -> URLs the frontend can load
        image_urls = {
            "address_image": "/outputs/visuals/" + os.path.basename(paths["address_image"]),
            "top_image": "/outputs/visuals/" + os.path.basename(paths["top_image"]),
            "impact_image": "/outputs/visuals/" + os.path.basename(paths["impact_image"]),
        }

        return JSONResponse(
            {
                "pose_meta": pose.get("meta", {}),
                "phases": phases,
                "image_paths": paths,
                "images": image_urls,
                "ball_xy": list(ball_xy) if ball_xy is not None else None,
                "note": "Images are saved on disk. Next upgrade: return base64 or an annotated video.",
            }
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.post("/analyze/full")
async def analyze_full(
    video: UploadFile = File(...),
    ball_x: Optional[float] = Form(None),
    ball_y: Optional[float] = Form(None),
    view: str = "side_on",
    handedness: str = "right",
):
    """
    Upload once, get all results streamed back as Server-Sent Events.

    Events (in order):
      pose      — pose extraction complete, meta sent
      phases    — swing phases detected
      report    — coaching report ready  ← frontend can render feedback here
      visuals   — 3 keyframe images saved  ← frontend can show images here
      annotated — annotated video saved  ← frontend can play video here
      done      — stream finished
      error / visuals_error / annotated_error — partial failure with message
    """
    suffix = os.path.splitext(video.filename or "upload.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await video.read()
        tmp.write(content)

    ball_xy = (float(ball_x), float(ball_y)) if (ball_x is not None and ball_y is not None) else None

    filename_base = os.path.splitext(video.filename or "swing")[0]
    filename_base = "".join([c for c in filename_base if c.isalnum() or c in ("-", "_")])[:40] or "swing"

    async def event_stream():
        def sse(event: str, data: Any) -> str:
            return f"data: {json.dumps({'event': event, 'data': data})}\n\n"

        try:
            pose = extract_pose_to_dict(tmp_path, view=view)
            yield sse("pose", pose["meta"])

            phases = detect_swing_phases(pose, ball_xy=ball_xy)
            yield sse("phases", phases)

            metrics = compute_golf_metrics(pose, phases, handedness=handedness)
            feedback = generate_ai_coaching(metrics, handedness=handedness) or generate_coaching_feedback(metrics)
            db = load_reference_db()
            ref_score = score_against_reference(extract_phase_vectors(pose, phases, handedness), db)
            report = {"phases": phases, "metrics": metrics, "feedback": feedback}
            yield sse("report", {
                "report": report,
                "pose_meta": pose.get("meta", {}),
                "ball_xy": list(ball_xy) if ball_xy else None,
                "reference_score": ref_score,
            })

            if phases.get("ok"):
                prefix = f"{filename_base}_{phases['address']['frame']}_{phases['impact']['frame']}"
                out_dir = OUTPUTS_DIR / "visuals"

                try:
                    paths = save_keyframe_images(pose, phases, out_dir=out_dir, prefix=prefix)
                    image_urls = {
                        "address_image": "/outputs/visuals/" + os.path.basename(paths["address_image"]),
                        "top_image": "/outputs/visuals/" + os.path.basename(paths["top_image"]),
                        "impact_image": "/outputs/visuals/" + os.path.basename(paths["impact_image"]),
                    }
                    yield sse("visuals", {
                        "images": image_urls,
                        "phases": phases,
                        "pose_meta": pose.get("meta", {}),
                        "ball_xy": list(ball_xy) if ball_xy else None,
                        "note": "Keyframe images with pose overlay.",
                    })
                except Exception as e:
                    yield sse("visuals_error", {"error": str(e)})

                try:
                    result = save_annotated_video(
                        pose, phases,
                        out_dir=out_dir,
                        prefix=prefix,
                        include_phase_labels=True,
                    )
                    annotated_url = "/outputs/visuals/" + os.path.basename(result["video"])
                    yield sse("annotated", {
                        "annotated": result,
                        "annotated_url": annotated_url,
                        "phases": phases,
                        "pose_meta": pose.get("meta", {}),
                        "ball_xy": list(ball_xy) if ball_xy else None,
                        "note": "Annotated video with pose overlay on every frame.",
                    })
                except Exception as e:
                    yield sse("annotated_error", {"error": str(e)})

            yield sse("done", {})

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/analyze/annotated")
async def analyze_annotated(
    video: UploadFile = File(...),
    ball_x: Optional[float] = Form(None),
    ball_y: Optional[float] = Form(None),
    view: str = "side_on",
    handedness: str = "right",
    include_phase_labels: bool = False,
):
    """
    Upload a golf swing video and save an annotated MP4 with pose overlay on every frame.
    Optionally labels the key frames (ADDRESS/TOP/IMPACT).

    Returns a file path to the saved annotated video (v1).
    """
    suffix = os.path.splitext(video.filename or "upload.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await video.read()
        tmp.write(content)

    try:
        pose = extract_pose_to_dict(tmp_path, view=view)
        ball_xy = (float(ball_x), float(ball_y)) if (ball_x is not None and ball_y is not None) else None
        phases = detect_swing_phases(pose, ball_xy=ball_xy)

        # Save video under backend/outputs/visuals
        out_dir = OUTPUTS_DIR / "visuals"

        # Prefix based on original filename (safe-ish) or a fallback
        base = os.path.splitext(video.filename or "swing")[0]
        base = "".join([c for c in base if c.isalnum() or c in ("-", "_")])[:40] or "swing"
        if phases.get("ok"):
            prefix = f"{base}_{phases['address']['frame']}_{phases['impact']['frame']}"
        else:
            prefix = f"{base}_annotated"

        result = save_annotated_video(
            pose,
            phases,
            out_dir=out_dir,
            prefix=prefix,
            include_phase_labels=include_phase_labels,
        )

        # Convert filesystem path -> URL the frontend can load
        annotated_url = "/outputs/visuals/" + os.path.basename(result["video"])

        return JSONResponse(
            {
                "pose_meta": pose.get("meta", {}),
                "phases": phases,
                "annotated": result,
                "annotated_url": annotated_url,
                "ball_xy": list(ball_xy) if ball_xy is not None else None,
                "note": "Annotated video is saved on disk. Next upgrade: stream or return a URL/base64 if needed.",
            }
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── Reference swing library ───────────────────────────────────────────────────

@app.post("/reference/add")
async def reference_add(
    video: UploadFile = File(...),
    label: str = Form("reference"),
    handedness: str = Form("right"),
    view: str = Form("side_on"),
):
    """Upload a reference swing video and add its pose snapshot to the reference DB."""
    suffix = os.path.splitext(video.filename or "upload.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await video.read()
        tmp.write(content)

    try:
        pose = extract_pose_to_dict(tmp_path, view=view)
        phases = detect_swing_phases(pose)
        db = load_reference_db()
        added = add_swing_to_db(db, pose, phases, handedness=handedness, label=label)
        if added:
            save_reference_db(db)
        return JSONResponse({
            "added": added,
            "label": label,
            "n_references": len(db.get("swings", [])),
            "phases_ok": phases.get("ok", False),
            "message": "Added to reference library." if added else "Impact phase could not be detected — not added.",
        })
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.get("/reference/stats")
def reference_stats():
    """Return metadata about the current reference swing database."""
    db = load_reference_db()
    swings = db.get("swings", [])
    return JSONResponse({
        "n_references": len(swings),
        "labels": [s.get("label", "reference") for s in swings],
    })


@app.delete("/reference/clear")
def reference_clear():
    """Wipe the reference swing database."""
    save_reference_db({"version": 1, "swings": []})
    return JSONResponse({"cleared": True, "n_references": 0})
