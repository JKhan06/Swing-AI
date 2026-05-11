"""Claude-powered coaching and chat for SwingAI."""
import json
from typing import Any, AsyncGenerator, Dict, List

import anthropic

COACHING_SYSTEM = """\
You are an expert golf coach analyzing a player's swing from computer vision biomechanics data.
Give 5-6 specific, actionable coaching tips. Be encouraging but honest.

Return ONLY a valid JSON array — no markdown fences, no explanation outside the array:
[{"title": "Category", "status": "good|ok|needs_work|unknown", "tip": "Specific advice."}]"""


def _fmt(v: Any, digits: int = 2, suffix: str = "") -> str:
    if v is None:
        return "not measured"
    return f"{round(float(v), digits)}{suffix}"


def generate_ai_coaching(metrics: Dict[str, Any], handedness: str = "right") -> List[Dict[str, Any]]:
    """Generate coaching tips via Claude. Returns [] on any failure so caller can fall back."""
    if not metrics.get("ok"):
        return []

    tempo = metrics.get("tempo_ratio")
    tempo_note = ""
    if tempo is not None:
        t = float(tempo)
        if 2.6 <= t <= 3.4:
            tempo_note = " (ideal range)"
        elif t < 2.6:
            tempo_note = " (downswing may be rushed)"
        else:
            tempo_note = " (very slow backswing tempo)"

    prompt = f"""{handedness.capitalize()}-handed golfer — swing metrics from computer vision:

Tempo: {_fmt(tempo, 2)}:1{tempo_note}
  Backswing: {_fmt(metrics.get('backswing_sec'), 3)}s | Downswing: {_fmt(metrics.get('downswing_sec'), 3)}s

Head sway (normalized): {_fmt(metrics.get('head_sway_norm'), 4)}
  (±0.06 = excellent · ±0.12 = acceptable · >0.12 = too much lateral movement)

Hip sway (normalized): {_fmt(metrics.get('hip_sway_norm'), 4)}
  (same scale as head sway)

Spine angle change address→impact: {_fmt(metrics.get('spine_angle_change_deg'), 1)}°
  (<4° = excellent · 4–8° = acceptable · >8° = early extension)

Shoulder turn address→top: {_fmt(metrics.get('shoulder_turn_proxy_deg'), 1)}°
  (>35° = excellent · 25–35° = acceptable · <25° = restricted rotation)

Lead wrist angle at impact: {_fmt(metrics.get('lead_wrist_angle_deg'), 1)}°
  (experimental — higher = more released at impact)

Return the JSON array now."""

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=COACHING_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # Strip markdown code fences if model adds them despite instructions
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return []


CHAT_SYSTEM_BASE = """\
You are an expert golf coach. A golfer just had their swing analyzed by computer vision software.
You have their biomechanical metrics and the coaching feedback they received.
Answer their questions conversationally, referencing their specific numbers where relevant.
Be encouraging, direct, and practical. Keep responses concise — 2–4 sentences unless a detailed
explanation is clearly warranted."""


def _build_context_block(swing_context: Dict[str, Any]) -> str:
    metrics = swing_context.get("metrics") or {}
    phases = swing_context.get("phases") or {}
    feedback = swing_context.get("feedback") or []
    handedness = swing_context.get("handedness", "right")

    lines = [f"Golfer: {handedness}-handed"]

    if phases.get("ok"):
        pm = phases.get("metrics") or {}
        lines.append(
            f"Tempo: {pm.get('tempo_ratio')}:1 | "
            f"Backswing: {pm.get('backswing_sec')}s | "
            f"Downswing: {pm.get('downswing_sec')}s"
        )

    if metrics.get("ok"):
        lines += [
            f"Head sway (norm): {metrics.get('head_sway_norm')}",
            f"Hip sway (norm): {metrics.get('hip_sway_norm')}",
            f"Spine angle change: {metrics.get('spine_angle_change_deg')}°",
            f"Shoulder turn: {metrics.get('shoulder_turn_proxy_deg')}°",
            f"Lead wrist angle: {metrics.get('lead_wrist_angle_deg')}°",
        ]

    if feedback:
        lines.append("\nCoaching feedback already given:")
        for fb in feedback:
            lines.append(f"  [{fb.get('status', '?')}] {fb.get('title')}: {fb.get('tip')}")

    return "\n".join(lines)


async def stream_chat(
    message: str,
    swing_context: Dict[str, Any],
    history: List[Dict[str, str]],
) -> AsyncGenerator[str, None]:
    """Async generator yielding text chunks from Claude for a chat message."""
    context_block = _build_context_block(swing_context)
    system = CHAT_SYSTEM_BASE + "\n\nSwing analysis context:\n" + context_block

    messages = []
    for h in history:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    client = anthropic.AsyncAnthropic()
    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system,
        messages=messages,
    ) as s:
        async for text in s.text_stream:
            yield text
