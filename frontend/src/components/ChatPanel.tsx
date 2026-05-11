"use client";

import { useEffect, useRef, useState } from "react";
import { sendChatMessage, type ChatMessage, type SwingContext } from "@/lib/api";

interface ChatPanelProps { swingContext: SwingContext }

const SUGGESTED = [
  "What should I work on first?",
  "Explain my tempo ratio",
  "How do I fix early extension?",
  "Is my shoulder turn good?",
];

export default function ChatPanel({ swingContext }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput]       = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async (text: string) => {
    text = text.trim();
    if (!text || isStreaming) return;
    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages(prev => [...prev, userMsg, { role: "assistant", content: "" }]);
    setInput("");
    setIsStreaming(true);
    try {
      await sendChatMessage(text, swingContext, [...messages, userMsg], (chunk) => {
        setMessages(prev => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === "assistant") next[next.length - 1] = { ...last, content: last.content + chunk };
          return next;
        });
      });
    } catch {
      setMessages(prev => {
        const next = [...prev];
        next[next.length - 1] = { role: "assistant", content: "Sorry, something went wrong. Please try again." };
        return next;
      });
    } finally {
      setIsStreaming(false);
      inputRef.current?.focus();
    }
  };

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800">
      <div className="px-4 pt-4 pb-2 border-b border-slate-800">
        <p className="text-[10px] text-slate-500 uppercase tracking-widest font-medium">AI Coach</p>
        <p className="text-xs text-slate-500 mt-0.5">Ask anything about your swing</p>
      </div>

      {/* Thread */}
      <div className="h-56 overflow-y-auto px-4 py-3 space-y-2.5">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2.5">
            <p className="text-xs text-slate-600">Try asking:</p>
            <div className="flex flex-wrap gap-1.5 justify-center">
              {SUGGESTED.map(q => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="text-[11px] px-2.5 py-1 rounded-full border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-300 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[75%] px-3 py-2 rounded-xl text-xs leading-relaxed ${
                msg.role === "user"
                  ? "bg-green-600/90 text-white rounded-br-sm"
                  : "bg-slate-800 border border-slate-700 text-slate-300 rounded-bl-sm"
              }`}>
                {msg.content || (isStreaming && i === messages.length - 1 ? (
                  <span className="inline-flex gap-1 items-center h-3">
                    <span className="w-1 h-1 rounded-full bg-slate-500 animate-bounce" />
                    <span className="w-1 h-1 rounded-full bg-slate-500 animate-bounce [animation-delay:0.15s]" />
                    <span className="w-1 h-1 rounded-full bg-slate-500 animate-bounce [animation-delay:0.3s]" />
                  </span>
                ) : null)}
              </div>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-4 pb-4 pt-2 border-t border-slate-800 flex gap-2">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }}
          placeholder="Ask about your swing…"
          disabled={isStreaming}
          className="flex-1 px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-green-600 disabled:opacity-50 transition"
        />
        <button
          onClick={() => send(input)}
          disabled={!input.trim() || isStreaming}
          className="px-4 py-1.5 bg-green-600 text-white rounded-lg text-xs font-medium hover:bg-green-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-colors"
        >
          {isStreaming ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
