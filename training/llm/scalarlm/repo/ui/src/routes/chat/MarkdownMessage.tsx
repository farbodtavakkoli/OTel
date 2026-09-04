import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownMessageProps {
  content: string;
}

/**
 * Minimal markdown renderer for chat bubbles. No runtime syntax highlighting
 * (adds multi-MB for the highlighter); code blocks are rendered as `<pre>`
 * with neutral styling — readable enough for v1. See docs/ui-design.md §6.5.
 *
 * Memoised because react-markdown re-parses on every prop change and assistant
 * messages update many times per second during streaming.
 */
export const MarkdownMessage = memo(function MarkdownMessage({
  content,
}: MarkdownMessageProps) {
  return (
    <div className="prose-chat">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children }) {
            const isBlock = (className ?? "").includes("language-");
            if (isBlock) {
              return (
                <pre className="my-2 overflow-x-auto rounded-md border border-border-subtle bg-bg/70 px-3 py-2 font-mono text-[12px] leading-relaxed text-fg">
                  <code>{children}</code>
                </pre>
              );
            }
            return (
              <code className="rounded bg-bg/70 px-1 py-0.5 font-mono text-[12px] text-fg">
                {children}
              </code>
            );
          },
          a({ href, children }) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="text-accent underline underline-offset-2 hover:text-accent-hover"
              >
                {children}
              </a>
            );
          },
          p({ children }) {
            return <p className="my-1.5 leading-relaxed">{children}</p>;
          },
          ul({ children }) {
            return <ul className="my-1.5 list-disc pl-6">{children}</ul>;
          },
          ol({ children }) {
            return <ol className="my-1.5 list-decimal pl-6">{children}</ol>;
          },
          table({ children }) {
            return (
              <div className="my-2 overflow-x-auto rounded-md border border-border-subtle">
                <table className="w-full border-collapse text-sm">
                  {children}
                </table>
              </div>
            );
          },
          th({ children }) {
            return (
              <th className="border-b border-border-subtle bg-bg/50 px-3 py-1.5 text-left font-medium text-fg-muted">
                {children}
              </th>
            );
          },
          td({ children }) {
            return (
              <td className="border-b border-border-subtle px-3 py-1.5">
                {children}
              </td>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
