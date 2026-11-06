import type { Config } from "tailwindcss";

const rgbVar = (name: string) => `rgb(var(${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        bg: {
          DEFAULT: rgbVar("--color-bg"),
          card: rgbVar("--color-bg-card"),
          hover: rgbVar("--color-bg-hover"),
        },
        border: {
          DEFAULT: rgbVar("--color-border"),
          subtle: rgbVar("--color-border-subtle"),
        },
        fg: {
          DEFAULT: rgbVar("--color-fg"),
          muted: rgbVar("--color-fg-muted"),
          subtle: rgbVar("--color-fg-subtle"),
        },
        accent: {
          DEFAULT: rgbVar("--color-accent"),
          hover: rgbVar("--color-accent-hover"),
        },
        success: {
          DEFAULT: rgbVar("--color-success"),
          hover: rgbVar("--color-success-hover"),
        },
        warning: {
          DEFAULT: rgbVar("--color-warning"),
          hover: rgbVar("--color-warning-hover"),
        },
        danger: {
          DEFAULT: rgbVar("--color-danger"),
          hover: rgbVar("--color-danger-hover"),
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
