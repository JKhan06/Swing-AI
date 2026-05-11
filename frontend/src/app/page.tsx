"use client";

import { useState, useRef, type ChangeEvent, type DragEvent } from "react";
import {
  analyzeReport,
  analyzeVisuals,
  analyzeAnnotated,
  analyzeStream,
  type ReportResponse,
  type VisualsResponse,
  type AnnotatedResponse,
  type SwingContext,
  type ReferenceScore,
} from "@/lib/api";

import ResultCard         from "@/components/ResultCard";
import MetricsPanel       from "@/components/MetricsPanel";
import ChatPanel          from "@/components/ChatPanel";
import ReferenceScoreCard from "@/components/ReferenceScore";
import ReferenceLibrary   from "@/components/ReferenceLibrary";

const BACKEND = "http://127.0.0.1:8000";

type SessionRun = {
  id: string;
  createdAt: number;
  fileName: string;
  handedness: string;
  reportData: ReportResponse | null;
  visualsData: VisualsResponse | null;
  annotatedData: AnnotatedResponse | null;
};

type FullStep = "extracting" | "report" | "visuals" | "annotated" | "done" | null;

const STEP_LABEL: Record<NonNullable<FullStep>, string> = {
  extracting: "Extracting pose…",
  report:     "Generating report…",
  visuals:    "Saving key frames…",
  annotated:  "Rendering video…",
  done:       "Done",
};

export default function Home() {
  const [file, setFile]           = useState<File | null>(null);
  const [dragging, setDragging]   = useState(false);
  const [handedness, setHandedness] = useState("right");
  const [loading, setLoading]     = useState<"report" | "visuals" | "annotated" | "full" | null>(null);
  const [fullStep, setFullStep]   = useState<FullStep>(null);
  const [error, setError]         = useState<string | null>(null);

  const [reportData, setReportData]     = useState<ReportResponse | null>(null);
  const [visualsData, setVisualsData]   = useState<VisualsResponse | null>(null);
  const [annotatedData, setAnnotatedData] = useState<AnnotatedResponse | null>(null);
  const [refScore, setRefScore]         = useState<ReferenceScore | null>(null);

  const [history, setHistory]     = useState<SessionRun[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [ballXY, setBallXY]       = useState<{ x: number; y: number } | null>(null);
  const [addrNatural, setAddrNatural] = useState<{ w: number; h: number } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── helpers ──────────────────────────────────────────────────────────────
  const clearResults = () => {
    setReportData(null); setVisualsData(null);
    setAnnotatedData(null); setRefScore(null);
    setBallXY(null); setAddrNatural(null);
  };

  const acceptFile = (f: File) => { setFile(f); setError(null); clearResults(); };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("video/")) acceptFile(f);
  };

  const addToHistory = (run: SessionRun) =>
    setHistory(prev => [run, ...prev].slice(0, 5));

  const loadFromHistory = (run: SessionRun) => {
    setError(null);
    setReportData(run.reportData); setVisualsData(run.visualsData);
    setAnnotatedData(run.annotatedData); setRefScore(null);
    setHandedness(run.handedness);
    setBallXY(null); setAddrNatural(null);
  };

  // ── analysis ─────────────────────────────────────────────────────────────
  const handleFullAnalysis = async () => {
    if (!file) return;
    clearResults();
    setLoading("full"); setFullStep("extracting"); setError(null);

    const runId = `${Date.now()}`, ts = Date.now();
    let report: ReportResponse | null = null;
    let visuals: VisualsResponse | null = null;
    let annotated: AnnotatedResponse | null = null;

    try {
      await analyzeStream(file, handedness, ballXY, (ev) => {
        switch (ev.event) {
          case "pose":     setFullStep("report"); break;
          case "report":
            report = ev.data as ReportResponse;
            setReportData(report);
            if (ev.data?.reference_score) setRefScore(ev.data.reference_score as ReferenceScore);
            setFullStep("visuals");
            break;
          case "visuals":
            visuals = ev.data as VisualsResponse;
            setVisualsData(visuals);
            setFullStep("annotated");
            break;
          case "annotated":
            annotated = ev.data as AnnotatedResponse;
            setAnnotatedData(annotated);
            setFullStep("done");
            break;
          case "error":
            setError(ev.error ?? "Server error");
            break;
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze swing");
    } finally {
      setLoading(null);
      addToHistory({ id: runId, createdAt: ts, fileName: file.name, handedness, reportData: report, visualsData: visuals, annotatedData: annotated });
    }
  };

  const handleReport = async () => {
    if (!file) return;
    setLoading("report"); setError(null);
    try {
      const r = await analyzeReport(file, handedness, ballXY);
      setReportData(r);
      if (r.reference_score) setRefScore(r.reference_score);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    } finally { setLoading(null); }
  };

  const handleVisuals = async () => {
    if (!file) return;
    setLoading("visuals"); setError(null);
    try { setVisualsData(await analyzeVisuals(file, handedness, ballXY)); }
    catch (err) { setError(err instanceof Error ? err.message : "Failed"); }
    finally { setLoading(null); }
  };

  const handleAnnotated = async () => {
    if (!file) return;
    setLoading("annotated"); setError(null);
    try { setAnnotatedData(await analyzeAnnotated(file, handedness, ballXY)); }
    catch (err) { setError(err instanceof Error ? err.message : "Failed"); }
    finally { setLoading(null); }
  };

  const handleAddressClick = (e: React.MouseEvent<HTMLImageElement>) => {
    const img = e.currentTarget, rect = img.getBoundingClientRect();
    setBallXY({
      x: Math.round(((e.clientX - rect.left) / rect.width) * img.naturalWidth),
      y: Math.round(((e.clientY - rect.top)  / rect.height) * img.naturalHeight),
    });
  };

  const hasResults = reportData || visualsData || annotatedData;
  const swingContext: SwingContext | null = reportData
    ? { metrics: reportData.report.metrics, phases: reportData.report.phases, feedback: reportData.report.feedback, handedness: reportData.report.metrics.handedness }
    : null;

  return (
    <div className="space-y-4">

      {/* ── Upload card ─────────────────────────────────────────────────── */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
        <div className="flex flex-col sm:flex-row gap-3">

          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`flex-1 border border-dashed rounded-lg px-4 py-3 flex items-center gap-3 cursor-pointer transition-colors ${
              dragging  ? "border-green-500 bg-green-500/5" :
              file      ? "border-green-600/50 bg-slate-800/60" :
                          "border-slate-700 hover:border-slate-600 hover:bg-slate-800/40"
            }`}
          >
            <input ref={fileInputRef} type="file" accept="video/*,.mp4"
              onChange={(e: ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; if (f) acceptFile(f); }}
              className="hidden"
            />
            <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center shrink-0 text-base">
              {file ? "🎬" : "📂"}
            </div>
            {file ? (
              <div className="min-w-0">
                <p className="text-sm font-medium text-slate-200 truncate">{file.name}</p>
                <p className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(1)} MB · click to change</p>
              </div>
            ) : (
              <div>
                <p className="text-sm text-slate-400">Drop swing video here</p>
                <p className="text-xs text-slate-600">or click to browse · MP4 recommended</p>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex sm:flex-col gap-2 shrink-0">
            <select
              value={handedness}
              onChange={e => setHandedness(e.target.value)}
              disabled={loading !== null}
              className="flex-1 sm:flex-none px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-green-600 disabled:opacity-50"
            >
              <option value="right">Right-handed</option>
              <option value="left">Left-handed</option>
            </select>

            <button
              onClick={handleFullAnalysis}
              disabled={!file || loading !== null}
              className="flex-1 sm:flex-none px-5 py-1.5 bg-green-600 hover:bg-green-500 text-white text-sm font-semibold rounded-lg disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-colors"
            >
              {loading === "full" ? "Analyzing…" : "Analyze Swing"}
            </button>
          </div>
        </div>

        {/* Progress + advanced buttons row */}
        <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-800">
          <div className="flex items-center gap-2 min-h-[18px]">
            {loading === "full" && fullStep && (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs text-slate-500">{STEP_LABEL[fullStep]}</span>
              </>
            )}
          </div>
          <div className="flex gap-1.5">
            {[
              { label: "Report",   handler: handleReport,    k: "report"    },
              { label: "Frames",   handler: handleVisuals,   k: "visuals"   },
              { label: "Video",    handler: handleAnnotated, k: "annotated" },
            ].map(({ label, handler, k }) => (
              <button
                key={k}
                onClick={handler}
                disabled={!file || loading !== null}
                className="px-2.5 py-1 text-[11px] text-slate-500 border border-slate-800 rounded-md hover:bg-slate-800 hover:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                {loading === k ? "…" : label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Error ───────────────────────────────────────────────────────── */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 flex gap-2.5 items-start">
          <span className="text-red-400 text-sm shrink-0 mt-0.5">⚠</span>
          <div>
            <p className="text-red-300 font-semibold text-sm">Analysis failed</p>
            <p className="text-red-400/80 text-xs mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* ── Results ─────────────────────────────────────────────────────── */}
      {hasResults && (
        <div className="space-y-4">

          {/* Row 1: Metrics (wider) + Reference score (narrower) */}
          {(reportData || refScore) && (
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
              {reportData && (
                <div className="lg:col-span-3">
                  <MetricsPanel metrics={reportData.report.metrics} phases={reportData.report.phases} />
                </div>
              )}
              {refScore && (
                <div className="lg:col-span-2">
                  <ReferenceScoreCard score={refScore} />
                </div>
              )}
            </div>
          )}

          {/* Row 2: Coaching feedback — 2-col grid */}
          {reportData?.report.feedback && reportData.report.feedback.length > 0 && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-3 font-medium">Coaching Feedback</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {reportData.report.feedback.map((item, idx) => (
                  <ResultCard key={idx} title={item.title} status={item.status} tip={item.tip} />
                ))}
              </div>
            </div>
          )}

          {/* Row 3: Chat */}
          {swingContext && (
            <ChatPanel key={reportData?.pose_meta.video} swingContext={swingContext} />
          )}

          {/* Row 4: Key frames */}
          {visualsData && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <p className="text-[10px] text-slate-500 uppercase tracking-widest font-medium">Key Frames</p>
                {ballXY ? (
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] text-green-400 bg-green-500/10 border border-green-500/20 px-2 py-0.5 rounded-full">
                      Ball ({ballXY.x}, {ballXY.y})
                    </span>
                    <button onClick={() => setBallXY(null)} className="text-[11px] text-slate-500 hover:text-slate-300">
                      clear
                    </button>
                  </div>
                ) : (
                  <span className="text-[11px] text-slate-600">Click ball in Address frame to improve impact detection</span>
                )}
              </div>

              <div className="grid grid-cols-3 gap-3">
                {([
                  { label: "Address", key: "address_image" as const, clickable: true },
                  { label: "Top",     key: "top_image"     as const, clickable: false },
                  { label: "Impact",  key: "impact_image"  as const, clickable: false },
                ] as const).map(({ label, key, clickable }) => (
                  <div key={label}>
                    <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1.5">{label}</p>
                    <div className="relative">
                      <img
                        src={`${BACKEND}${visualsData.images?.[key] ?? ""}`}
                        alt={label}
                        className={`w-full rounded-lg border border-slate-800 ${clickable ? "cursor-crosshair" : ""}`}
                        onClick={clickable ? handleAddressClick : undefined}
                        onLoad={e => {
                          if (clickable) {
                            const el = e.currentTarget;
                            setAddrNatural({ w: el.naturalWidth, h: el.naturalHeight });
                          }
                        }}
                      />
                      {clickable && ballXY && addrNatural && (
                        <div style={{
                          position: "absolute",
                          left: `${(ballXY.x / addrNatural.w) * 100}%`,
                          top:  `${(ballXY.y / addrNatural.h) * 100}%`,
                          transform: "translate(-50%,-50%)",
                          width: 12, height: 12,
                          borderRadius: "50%",
                          border: "2px solid white",
                          boxShadow: "0 0 0 2px rgba(0,0,0,0.7)",
                          background: "rgba(239,68,68,0.9)",
                          pointerEvents: "none",
                        }} />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Row 5: Annotated video */}
          {annotatedData && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <p className="text-[10px] text-slate-500 uppercase tracking-widest font-medium">Annotated Video</p>
                <div className="flex gap-3 text-[11px] text-slate-600">
                  <span>{annotatedData.annotated.frame_count_written} frames</span>
                  <span>{annotatedData.annotated.fps.toFixed(0)} fps</span>
                  <span>{annotatedData.annotated.width}×{annotatedData.annotated.height}</span>
                </div>
              </div>
              {annotatedData.annotated_url ? (
                <video
                  controls playsInline
                  className="w-full rounded-lg border border-slate-800"
                  src={`${BACKEND}${annotatedData.annotated_url}`}
                />
              ) : (
                <p className="text-xs text-red-400">Missing video URL from server.</p>
              )}
            </div>
          )}

          {/* Raw JSON */}
          {reportData && (
            <details className="text-xs">
              <summary className="cursor-pointer text-slate-700 hover:text-slate-500 select-none py-1">
                Raw JSON
              </summary>
              <pre className="mt-2 p-3 bg-slate-950 border border-slate-800 text-green-500 rounded-lg text-[11px] overflow-auto max-h-64">
                {JSON.stringify(reportData, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}

      {/* ── Reference library ───────────────────────────────────────────── */}
      <ReferenceLibrary />

      {/* ── History ─────────────────────────────────────────────────────── */}
      {history.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
          <button
            onClick={() => setHistoryOpen(o => !o)}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition-colors"
          >
            <span className="text-sm font-medium text-slate-300">Recent Analyses</span>
            <span className="text-[11px] text-slate-600">
              {history.length} run{history.length > 1 ? "s" : ""} · {historyOpen ? "▲" : "▼"}
            </span>
          </button>

          {historyOpen && (
            <div className="border-t border-slate-800 divide-y divide-slate-800">
              {history.map(run => {
                const thumb = run.visualsData?.images?.impact_image ?? run.visualsData?.images?.address_image ?? null;
                return (
                  <div key={run.id} className="flex gap-3 px-4 py-3 items-start hover:bg-slate-800/30 transition-colors">
                    {thumb ? (
                      <img src={`${BACKEND}${thumb}`} alt="" className="w-14 h-14 object-cover rounded-lg border border-slate-800 shrink-0" />
                    ) : (
                      <div className="w-14 h-14 rounded-lg border border-slate-800 bg-slate-800 shrink-0" />
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-slate-300 truncate">{run.fileName}</p>
                      <p className="text-xs text-slate-600 mt-0.5">
                        {run.handedness === "right" ? "Right" : "Left"} · {new Date(run.createdAt).toLocaleTimeString()}
                      </p>
                      <div className="flex gap-2 mt-2">
                        <button onClick={() => loadFromHistory(run)}
                          className="px-2.5 py-1 text-xs rounded-md bg-green-600/20 text-green-400 hover:bg-green-600/30 border border-green-600/20 transition-colors">
                          Load
                        </button>
                        <button onClick={() => setHistory(prev => prev.filter(r => r.id !== run.id))}
                          className="px-2.5 py-1 text-xs rounded-md border border-slate-700 text-slate-500 hover:bg-slate-800 transition-colors">
                          Remove
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
