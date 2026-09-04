import { Link } from "react-router-dom";

import { BlackHoleCanvas } from "@/components/BlackHoleCanvas";

const cards = [
  {
    to: "/chat",
    title: "Chat",
    description: "Send prompts to the running model and stream responses.",
  },
  {
    to: "/train",
    title: "Train",
    description: "Submit datasets, watch loss curves, and tail training logs.",
  },
  {
    to: "/metrics",
    title: "Metrics",
    description: "Inference throughput, queue depth, and cluster capacity.",
  },
  {
    to: "/models",
    title: "Models",
    description: "Base model and every post-trained adapter on this deployment.",
  },
];

export function HomePage() {
  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <div className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight">ScalarLM</h1>
        <p className="mt-2 text-sm text-fg-muted">
          Closed-loop LLM experimentation. Run a model, post-train it, serve the
          result — all in the same deployment.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {cards.map((card) => (
          <Link
            key={card.to}
            to={card.to}
            className="group rounded-lg border border-border-subtle bg-bg-card p-4 transition-colors hover:border-border hover:bg-bg-hover"
          >
            <h2 className="text-base font-medium text-fg group-hover:text-accent">
              {card.title}
            </h2>
            <p className="mt-1 text-sm text-fg-muted">{card.description}</p>
          </Link>
        ))}
      </div>

      <figure className="mt-10 overflow-hidden rounded-lg border border-border-subtle bg-black">
        <div className="relative h-[520px] w-full">
          <BlackHoleCanvas className="absolute inset-0" />
          <figcaption className="pointer-events-none absolute left-3 top-3 font-mono text-[10px] uppercase tracking-[0.2em] text-white/40">
            drag · scroll to zoom
          </figcaption>
          <figcaption className="pointer-events-none absolute bottom-3 right-3 font-mono text-[10px] uppercase tracking-[0.25em] text-amber-200/40">
            gargantua · raymarched
          </figcaption>
        </div>
      </figure>
    </div>
  );
}
