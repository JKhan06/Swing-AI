import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SwingAI — Golf Swing Analysis",
  description: "AI-powered golf swing analysis",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-[#090d18] min-h-screen text-white antialiased">
        <header className="bg-slate-900/95 backdrop-blur-sm border-b border-slate-800 sticky top-0 z-50">
          <div className="max-w-5xl mx-auto px-4 h-12 flex items-center gap-3">
            <div className="w-7 h-7 bg-green-600 rounded-md flex items-center justify-center shrink-0">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a10 10 0 1 0 10 10" />
                <path d="M12 8v4l3 3" />
              </svg>
            </div>
            <span className="text-white font-bold text-sm tracking-tight">SwingAI</span>
            <span className="text-slate-600 text-xs hidden sm:inline">Golf Swing Analysis</span>
          </div>
        </header>
        <main className="max-w-5xl mx-auto px-4 py-5">
          {children}
        </main>
      </body>
    </html>
  );
}
