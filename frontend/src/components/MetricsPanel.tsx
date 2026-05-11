"use client";

import type { ReportResponse } from "@/lib/api";

type Metrics = ReportResponse["report"]["metrics"];
type Phases  = ReportResponse["report"]["phases"];

interface MetricsPanelProps {
  metrics: Metrics;
  phases: Phases;
}

type StatusType = "good" | "ok" | "needs_work" | "unknown";

function statusColor(s: StatusType) {
  return {
    good:       "text-green-400 bg-green-500/10 border-green-500/30",
    ok:         "text-amber-400 bg-amber-500/10 border-amber-500/30",
    needs_work: "text-red-400 bg-red-500/10 border-red-500/30",
    unknown:    "text-slate-500 bg-slate-700/40 border-slate-700",
  }[s];
}

function MetricRow({ label, value, status, hint }: { label: string; value: string; status: StatusType; hint?: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-800 last:border-0 gap-3">
      <div className="min-w-0">
        <span className="text-sm text-slate-300">{label}</span>
        {hint && <span className="text-xs text-slate-600 ml-1.5 hidden sm:inline">{hint}</span>}
      </div>
      <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border shrink-0 ${statusColor(status)}`}>
        {value}
      </span>
    </div>
  );
}

function tempoStatus(r: number | null): StatusType {
  if (r === null) return "unknown";
  return r >= 2.6 && r <= 3.4 ? "good" : r >= 2.0 && r <= 4.0 ? "ok" : "needs_work";
}
function swayStatus(n: number | null): StatusType {
  if (n === null) return "unknown";
  const a = Math.abs(n);
  return a <= 0.06 ? "good" : a <= 0.12 ? "ok" : "needs_work";
}
function spineStatus(d: number | null): StatusType {
  if (d === null) return "unknown";
  return d <= 4 ? "good" : d <= 8 ? "ok" : "needs_work";
}
function shoulderStatus(d: number | null): StatusType {
  if (d === null) return "unknown";
  return d >= 35 ? "good" : d >= 25 ? "ok" : "needs_work";
}

export default function MetricsPanel({ metrics, phases }: MetricsPanelProps) {
  const pm = phases?.metrics;

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
      {/* Phase timeline */}
      {phases?.ok && (
        <div className="px-4 pt-4 pb-3 border-b border-slate-800">
          <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-3 font-medium">Swing Timeline</p>
          <div className="flex items-center gap-0">
            {[
              { label: "Address", time: phases.address.time_sec },
              { label: "Top",     time: phases.top.time_sec },
              { label: "Impact",  time: phases.impact.time_sec },
            ].map((phase, i) => (
              <div key={phase.label} className="flex items-center flex-1">
                <div className="text-center flex-1">
                  <div className="w-2 h-2 rounded-full bg-green-500 mx-auto mb-1" />
                  <p className="text-white text-[11px] font-semibold">{phase.label}</p>
                  <p className="text-slate-500 text-[11px]">{phase.time.toFixed(2)}s</p>
                </div>
                {i < 2 && (
                  <div className="flex-1 flex flex-col items-center gap-0.5 px-1">
                    <div className="w-full h-px bg-slate-700" />
                    <p className="text-slate-600 text-[10px]">
                      {i === 0
                        ? `${pm?.backswing_sec?.toFixed(2)}s`
                        : `${pm?.downswing_sec?.toFixed(2)}s`}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
          {pm?.tempo_ratio && (
            <p className="text-center text-[11px] text-slate-500 mt-2">
              Tempo <span className="text-white font-semibold">{pm.tempo_ratio.toFixed(2)}:1</span>
              <span className="ml-1 text-slate-600">(ideal 2.6–3.4)</span>
            </p>
          )}
        </div>
      )}

      {/* Metrics rows */}
      {metrics?.ok && (
        <div className="px-4 py-2">
          <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 pt-1 font-medium">Body Metrics</p>
          <MetricRow
            label="Tempo"
            value={pm?.tempo_ratio != null ? `${pm.tempo_ratio.toFixed(2)}:1` : "—"}
            status={tempoStatus(pm?.tempo_ratio ?? null)}
            hint="ideal 2.6–3.4"
          />
          <MetricRow
            label="Head Stability"
            value={metrics.head_sway_norm != null ? `${(Math.abs(metrics.head_sway_norm) * 100).toFixed(1)}%` : "—"}
            status={swayStatus(metrics.head_sway_norm)}
            hint="lower is better"
          />
          <MetricRow
            label="Hip Sway"
            value={metrics.hip_sway_norm != null ? `${(Math.abs(metrics.hip_sway_norm) * 100).toFixed(1)}%` : "—"}
            status={swayStatus(metrics.hip_sway_norm)}
            hint="lower is better"
          />
          <MetricRow
            label="Spine Angle"
            value={metrics.spine_angle_change_deg != null ? `${metrics.spine_angle_change_deg.toFixed(1)}°` : "—"}
            status={spineStatus(metrics.spine_angle_change_deg)}
            hint="<4° great"
          />
          <MetricRow
            label="Shoulder Turn"
            value={metrics.shoulder_turn_proxy_deg != null ? `${metrics.shoulder_turn_proxy_deg.toFixed(1)}°` : "—"}
            status={shoulderStatus(metrics.shoulder_turn_proxy_deg)}
            hint=">35° great"
          />
        </div>
      )}
    </div>
  );
}
