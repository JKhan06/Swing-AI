"use client";

interface ResultCardProps {
  title: string;
  status: "good" | "ok" | "needs_work" | "unknown";
  tip: string;
}

const config = {
  good:       { border: "border-l-green-500",  badge: "bg-green-500/15 text-green-400 border border-green-500/30",  dot: "bg-green-500",  label: "Good" },
  ok:         { border: "border-l-amber-400",  badge: "bg-amber-500/15 text-amber-400 border border-amber-500/30",  dot: "bg-amber-400",  label: "OK" },
  needs_work: { border: "border-l-red-500",    badge: "bg-red-500/15 text-red-400 border border-red-500/30",        dot: "bg-red-500",    label: "Fix" },
  unknown:    { border: "border-l-slate-600",  badge: "bg-slate-700/50 text-slate-400 border border-slate-700",     dot: "bg-slate-500",  label: "—" },
};

export default function ResultCard({ title, status, tip }: ResultCardProps) {
  const c = config[status];
  return (
    <div className={`bg-slate-800/60 border border-slate-700/60 border-l-2 ${c.border} rounded-lg p-3.5`}>
      <div className="flex items-center justify-between mb-1.5 gap-2">
        <span className="font-semibold text-slate-100 text-sm leading-tight">{title}</span>
        <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full shrink-0 ${c.badge}`}>
          {c.label}
        </span>
      </div>
      <p className="text-slate-400 text-xs leading-relaxed">{tip}</p>
    </div>
  );
}
