const BASE_URL = "http://127.0.0.1:8000";

export type BallXY = { x: number; y: number };

export interface ReferenceScore {
  overall: number | null;
  phases: {
    address: number | null;
    top: number | null;
    impact: number | null;
  };
  n_references: number;
}

export interface ReportResponse {
  pose_meta: {
    video: string;
    fps: number;
    width: number;
    height: number;
    frame_count: number;
    [key: string]: any;
  };
  ball_xy?: [number, number] | null;
  report: {
    phases: {
      ok: boolean;
      wrist: string;
      address: { frame: number; time_sec: number };
      top: { frame: number; time_sec: number };
      impact: { frame: number; time_sec: number };
      metrics: {
        backswing_sec: number;
        downswing_sec: number;
        tempo_ratio: number | null;
        fps: number;
      };
    };
    metrics: {
      ok: boolean;
      handedness: string;
      tempo_ratio: number | null;
      backswing_sec: number | null;
      downswing_sec: number | null;
      head_sway_px: number | null;
      head_sway_norm: number | null;
      hip_sway_px: number | null;
      hip_sway_norm: number | null;
      spine_angle_change_deg: number | null;
      shoulder_turn_proxy_deg: number | null;
      lead_wrist_angle_deg: number | null;
      trail_wrist_angle_deg: number | null;
      scale_ref_shoulder_width_px: number | null;
      [key: string]: any;
    };
    feedback: Array<{
      title: string;
      status: "good" | "ok" | "needs_work" | "unknown";
      tip: string;
    }>;
  };
  reference_score?: ReferenceScore;
}

export interface VisualsResponse {
  pose_meta: any;
  phases: any;
  ball_xy?: [number, number] | null;
  images: {
    address_image: string;
    top_image: string;
    impact_image: string;
  };
  note: string;
}

export interface AnnotatedResponse {
  pose_meta: any;
  phases: any;
  ball_xy?: [number, number] | null;
  annotated: {
    video: string;
    frame_count_written: number;
    fps: number;
    width: number;
    height: number;
  };
  annotated_url?: string;
  note: string;
}

export async function analyzeReport(
  file: File,
  handedness: string = "right",
  ballXY: BallXY | null = null
): Promise<ReportResponse> {
  const formData = new FormData();
  formData.append("video", file);
  formData.append("handedness", handedness);
  formData.append("view", "side_on");
  if (ballXY) {
    formData.append("ball_x", String(ballXY.x));
    formData.append("ball_y", String(ballXY.y));
  }

  const response = await fetch(`${BASE_URL}/analyze/report`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Report analysis failed: ${response.status} ${errorText}`);
  }

  return response.json();
}

export async function analyzeVisuals(
  file: File,
  handedness: string = "right",
  ballXY: BallXY | null = null
): Promise<VisualsResponse> {
  const formData = new FormData();
  formData.append("video", file);
  formData.append("handedness", handedness);
  formData.append("view", "side_on");
  if (ballXY) {
    formData.append("ball_x", String(ballXY.x));
    formData.append("ball_y", String(ballXY.y));
  }

  const response = await fetch(`${BASE_URL}/analyze/visuals`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Visuals generation failed: ${response.status} ${errorText}`);
  }

  return response.json();
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface SwingContext {
  metrics: ReportResponse["report"]["metrics"];
  phases: ReportResponse["report"]["phases"];
  feedback: ReportResponse["report"]["feedback"];
  handedness: string;
}

export async function sendChatMessage(
  message: string,
  swingContext: SwingContext,
  history: ChatMessage[],
  onChunk: (text: string) => void,
): Promise<void> {
  const response = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      swing_context: swingContext,
      history,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Chat failed: ${response.status} ${errorText}`);
  }

  if (!response.body) throw new Error("No response body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const line = chunk.trim();
      if (!line.startsWith("data: ")) continue;
      try {
        const parsed = JSON.parse(line.slice(6));
        if (parsed.text) onChunk(parsed.text);
      } catch {
        // ignore malformed lines
      }
    }
  }
}

export type StreamEventType =
  | "pose"
  | "phases"
  | "report"
  | "visuals"
  | "annotated"
  | "done"
  | "error"
  | "visuals_error"
  | "annotated_error";

export interface StreamEvent {
  event: StreamEventType;
  data?: any;
  error?: string;
}

export async function analyzeStream(
  file: File,
  handedness: string = "right",
  ballXY: BallXY | null = null,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const formData = new FormData();
  formData.append("video", file);
  formData.append("handedness", handedness);
  formData.append("view", "side_on");
  if (ballXY) {
    formData.append("ball_x", String(ballXY.x));
    formData.append("ball_y", String(ballXY.y));
  }

  const response = await fetch(`${BASE_URL}/analyze/full`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Stream failed: ${response.status} ${errorText}`);
  }

  if (!response.body) throw new Error("Response body is empty");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const line = chunk.trim();
      if (!line.startsWith("data: ")) continue;
      try {
        const parsed = JSON.parse(line.slice(6)) as StreamEvent;
        onEvent(parsed);
      } catch {
        // ignore malformed SSE line
      }
    }
  }
}

export async function addReferenceSwing(
  file: File,
  label: string,
  handedness: string = "right",
): Promise<{ added: boolean; label: string; n_references: number; message: string }> {
  const formData = new FormData();
  formData.append("video", file);
  formData.append("label", label);
  formData.append("handedness", handedness);

  const res = await fetch(`${BASE_URL}/reference/add`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(`Reference add failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getReferenceStats(): Promise<{ n_references: number; labels: string[] }> {
  const res = await fetch(`${BASE_URL}/reference/stats`);
  if (!res.ok) throw new Error("Could not fetch reference stats");
  return res.json();
}

export async function clearReferenceDb(): Promise<void> {
  const res = await fetch(`${BASE_URL}/reference/clear`, { method: "DELETE" });
  if (!res.ok) throw new Error("Could not clear reference database");
}

export async function analyzeAnnotated(
  file: File,
  handedness: string = "right",
  ballXY: BallXY | null = null
): Promise<AnnotatedResponse> {
  const formData = new FormData();
  formData.append("video", file);
  formData.append("handedness", handedness);
  formData.append("view", "side_on");
  formData.append("include_phase_labels", "true");
  if (ballXY) {
    formData.append("ball_x", String(ballXY.x));
    formData.append("ball_y", String(ballXY.y));
  }

  const response = await fetch(`${BASE_URL}/analyze/annotated`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Annotated video generation failed: ${response.status} ${errorText}`);
  }

  return response.json();
}
