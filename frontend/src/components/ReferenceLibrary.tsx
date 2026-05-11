"use client";

import { useState, useEffect, useRef, type ChangeEvent } from "react";
import { addReferenceSwing, getReferenceStats, clearReferenceDb } from "@/lib/api";

export default function ReferenceLibrary() {
  const [open, setOpen]             = useState(false);
  const [stats, setStats]           = useState<{ n_references: number; labels: string[] } | null>(null);
  const [uploading, setUploading]   = useState(false);
  const [label, setLabel]           = useState("");
  const [handedness, setHandedness] = useState("right");
  const [status, setStatus]         = useState<{ ok: boolean; msg: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const fetchStats = async () => {
    try { setStats(await getReferenceStats()); } catch { /* ignore */ }
  };

  useEffect(() => { fetchStats(); }, []);

  const handleFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setStatus(null);
    try {
      const res = await addReferenceSwing(file, label.trim() || file.name.replace(/\.[^.]+$/, ""), handedness);
      setStatus({ ok: res.added, msg: res.message });
      await fetchStats();
    } catch (err) {
      setStatus({ ok: false, msg: err instanceof Error ? err.message : "Upload failed" });
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const handleClear = async () => {
    if (!confirm(`Remove all ${stats?.n_references} reference swings?`)) return;
    try {
      await clearReferenceDb();
      setStatus({ ok: true, msg: "Library cleared." });
      await fetchStats();
    } catch {
      setStatus({ ok: false, msg: "Failed to clear library." });
    }
  };

  const count = stats?.n_references ?? 0;

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition-colors"
      >
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-md bg-slate-800 flex items-center justify-center shrink-0">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
            </svg>
          </div>
          <span className="text-sm font-medium text-slate-300">Reference Library</span>
          <span className={`text-xs px-1.5 py-0.5 rounded-full font-medium ${
            count > 0 ? "bg-green-500/15 text-green-400" : "bg-slate-800 text-slate-600"
          }`}>
            {count}
          </span>
        </div>
        <span className="text-[11px] text-slate-600">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="border-t border-slate-800 px-4 py-4 space-y-4">
          <p className="text-xs text-slate-500 leading-relaxed">
            Upload pro or high-quality swing videos. Each is processed once and stored as a pose snapshot.
            Your swings are then scored against this library.
          </p>

          <div className="space-y-2">
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Label (e.g. Rory McIlroy)"
                value={label}
                onChange={e => setLabel(e.target.value)}
                className="flex-1 px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-green-600"
              />
              <select
                value={handedness}
                onChange={e => setHandedness(e.target.value)}
                className="px-2 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-green-600"
              >
                <option value="right">Right</option>
                <option value="left">Left</option>
              </select>
            </div>
            <input ref={fileRef} type="file" accept="video/*,.mp4" onChange={handleFile} className="hidden" />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
              className="w-full py-2 border border-dashed border-slate-700 rounded-lg text-xs text-slate-500 hover:border-green-600 hover:text-green-500 hover:bg-green-500/5 disabled:opacity-50 transition-colors"
            >
              {uploading ? "Processing…" : "+ Add reference swing"}
            </button>
          </div>

          {status && (
            <div className={`px-3 py-2 rounded-lg text-xs ${
              status.ok ? "bg-green-500/10 text-green-400 border border-green-500/20" : "bg-red-500/10 text-red-400 border border-red-500/20"
            }`}>
              {status.msg}
            </div>
          )}

          {stats && stats.labels.length > 0 && (
            <div>
              <p className="text-[10px] text-slate-600 uppercase tracking-widest mb-2">In library</p>
              <ul className="space-y-1">
                {stats.labels.map((lbl, i) => (
                  <li key={i} className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="w-1 h-1 rounded-full bg-green-500 shrink-0" />
                    {lbl}
                  </li>
                ))}
              </ul>
              <button
                onClick={handleClear}
                className="mt-3 text-xs text-red-500/70 hover:text-red-400 transition-colors"
              >
                Clear all
              </button>
            </div>
          )}

          {count === 0 && !status && (
            <p className="text-xs text-slate-700 italic">No reference swings yet.</p>
          )}
        </div>
      )}
    </div>
  );
}
