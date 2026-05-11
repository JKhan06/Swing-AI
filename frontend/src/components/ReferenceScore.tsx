"use client";

import type { ReferenceScore } from "@/lib/api";

interface Props { score: ReferenceScore }

function tier(n: number | null) {
  if (n === null) return { label: "—",          ring: "#334155", text: "text-slate-500" };
  if (n >= 80)    return { label: "Excellent",   ring: "#22c55e", text: "text-green-400" };
  if (n >= 60)    return { label: "Good",        ring: "#f59e0b", text: "text-amber-400" };
  if (n >= 40)    return { label: "Fair",        ring: "#f97316", text: "text-orange-400" };
  return           { label: "Needs work",        ring: "#ef4444", text: "text-red-400" };
}

function Gauge({ value, size = 80 }: { value: number | null; size?: number }) {
  const r = (size - 10) / 2;
  const circ = 2 * Math.PI * r;
  const pct  = value !== null ? Math.max(0, Math.min(100, value)) / 100 : 0;
  const t    = tier(value);

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: "rotate(-90deg)" }}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#1e293b" strokeWidth={7} />
      {value !== null && (
        <circle
          cx={size/2} cy={size/2} r={r}
          fill="none" stroke={t.ring} strokeWidth={7}
          strokeDasharray={`${pct * circ} ${circ}`}
          strokeLinecap="round"
        />
      )}
    </svg>
  );
}

function PhaseBar({ label, value }: { label: string; value: number | null }) {
  const t   = tier(value);
  const pct = value !== null ? Math.max(0, Math.min(100, value)) : 0;
  return (
    <div>
      <div className="flex justify-between items-baseline mb-1">
        <span className="text-[11px] text-slate-500">{label}</span>
        <span className={`text-[11px] font-semibold ${value !== null ? t.text : "text-slate-700"}`}>
          {value !== null ? Math.round(value) : "—"}
        </span>
      </div>
      <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: value !== null ? tier(value).ring : "transparent" }}
        />
      </div>
    </div>
  );
}

export default function ReferenceScoreCard({ score }: Props) {
  const t = tier(score.overall);

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[10px] text-slate-500 uppercase tracking-widest font-medium">Reference Match</p>
        <span className="text-[10px] text-slate-600">{score.n_references} ref{score.n_references !== 1 ? "s" : ""}</span>
      </div>

      {score.n_references === 0 ? (
        <p className="text-xs text-slate-600 italic">Add reference swings below to enable scoring.</p>
      ) : (
        <div className="flex items-center gap-4">
          <div className="relative shrink-0">
            <Gauge value={score.overall} size={80} />
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className={`text-lg font-bold leading-none ${score.overall !== null ? t.text : "text-slate-700"}`}>
                {score.overall !== null ? Math.round(score.overall) : "—"}
              </span>
              <span className="text-[9px] text-slate-600 mt-0.5">/100</span>
            </div>
          </div>

          <div className="flex-1 space-y-2.5">
            <PhaseBar label="Address" value={score.phases.address} />
            <PhaseBar label="Top"     value={score.phases.top} />
            <PhaseBar label="Impact"  value={score.phases.impact} />
            <p className={`text-xs font-semibold mt-1 ${t.text}`}>{t.label}</p>
          </div>
        </div>
      )}
    </div>
  );
}
