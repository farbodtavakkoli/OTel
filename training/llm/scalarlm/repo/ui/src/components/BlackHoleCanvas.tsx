/**
 * BlackHoleCanvas — a raymarched Schwarzschild black hole rendered on a
 * fullscreen quad via plain WebGL1. Shader ported from the reference
 * three.js sketch; we skip three.js here because the only thing it was
 * doing was setting up a quad + ShaderMaterial, which is ~40 lines of
 * GL. No new ~600 KB dependency for the home page.
 *
 * Interaction: drag to orbit, wheel to zoom, idle-spin after 2.5 s of
 * no input.
 */

import { useEffect, useRef } from "react";

const VERT_SRC = `
attribute vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
`;

const FRAG_SRC = `
precision highp float;
uniform vec2 uResolution;
uniform float uTime;
uniform vec3 uCamPos;
uniform mat3 uCamRot;

const float PI = 3.14159265359;
const float HORIZON = 1.0;
const float DISK_INNER = 2.2;
const float DISK_OUTER = 5.6;
const int   MAX_STEPS = 240;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
float noise2(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i),             hash21(i + vec2(1.0, 0.0)), u.x),
             mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 6; i++) { v += a * noise2(p); p *= 2.13; a *= 0.5; }
  return v;
}

vec3 starfield(vec3 dir) {
  vec2 uv = vec2(atan(dir.z, dir.x) / (2.0 * PI) + 0.5, dir.y * 0.5 + 0.5);
  vec3 col = vec3(0.0);
  for (int layer = 0; layer < 3; layer++) {
    float scale = 220.0 + float(layer) * 280.0;
    vec2 sUv = uv * vec2(scale * 2.0, scale);
    vec2 cell = floor(sUv);
    vec2 fc = fract(sUv);
    float h = hash21(cell + float(layer) * 17.3);
    float thresh = 0.965;
    if (h > thresh) {
      vec2 starPos = vec2(0.5) + (vec2(hash21(cell + 1.0), hash21(cell + 2.0)) - 0.5) * 0.7;
      float d = distance(fc, starPos);
      float bright = pow((h - thresh) / (1.0 - thresh), 3.5);
      float intensity = exp(-d * 55.0) * bright * (1.6 - float(layer) * 0.3);
      vec3 starCol = mix(vec3(0.65, 0.78, 1.0), vec3(1.0, 0.92, 0.7), hash21(cell + 5.0));
      col += starCol * intensity;
    }
  }
  float n  = fbm(uv * 4.0);
  float n2 = fbm(uv * 9.0 + 5.0);
  vec3 neb = mix(vec3(0.004, 0.002, 0.012), vec3(0.025, 0.008, 0.04), n) * (n2 * 0.6 + 0.2);
  col += neb;
  return col;
}

vec3 sampleDisk(vec3 hit, vec3 rayDir) {
  float r = length(hit.xz);
  if (r < DISK_INNER || r > DISK_OUTER) return vec3(0.0);

  float ang = atan(hit.z, hit.x);
  vec3 vOrb = vec3(-sin(ang), 0.0, cos(ang)) * sqrt(0.5 / r);
  float vDotN = dot(vOrb, -rayDir);
  float doppler = 1.0 / pow(max(1.0 - vDotN * 0.85, 0.15), 3.2);
  doppler = clamp(doppler, 0.25, 6.0);

  float omega = 0.85 / pow(r, 1.3);
  float swirl = ang - uTime * omega * 2.2;

  vec2 sp = vec2(swirl * 3.2, r * 0.7);
  float n  = fbm(sp);
  float n2 = fbm(sp * 2.4 + 11.0);
  float turb = pow(mix(n, n2, 0.55), 1.5) * 1.3 + 0.3;

  float inner = smoothstep(DISK_INNER, DISK_INNER + 0.45, r);
  float outer = smoothstep(DISK_OUTER, DISK_OUTER - 1.6, r);
  float profile = inner * outer;

  float temp = pow(DISK_INNER / max(r, DISK_INNER), 0.75);

  vec3 hot  = vec3(1.0, 0.97, 0.88);
  vec3 mid  = vec3(1.0, 0.55, 0.18);
  vec3 cool = vec3(0.42, 0.13, 0.04);
  vec3 color = mix(cool, mid, smoothstep(0.0, 0.5, temp));
  color = mix(color, hot, smoothstep(0.5, 1.0, temp));

  float intensity = profile * turb * temp * 5.0 * doppler;
  return color * intensity;
}

void main() {
  vec2 ndc = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
  vec3 rayCam = normalize(vec3(ndc, -1.4));
  vec3 dir = normalize(uCamRot * rayCam);
  vec3 pos = uCamPos;

  vec3 L = cross(pos, dir);
  float h2 = dot(L, L);

  vec3 accum = vec3(0.0);
  bool escaped = false;

  for (int i = 0; i < MAX_STEPS; i++) {
    float r2 = dot(pos, pos);
    float r  = sqrt(r2);
    if (r < HORIZON * 1.005) break;
    if (r > 60.0) { escaped = true; break; }

    vec3 a = -1.5 * h2 * pos / (r2 * r2 * r);
    float ds = mix(0.04, 0.4, smoothstep(1.2, 14.0, r));

    vec3 newPos = pos + dir * ds + 0.5 * a * ds * ds;
    vec3 newDir = normalize(dir + a * ds);

    if (pos.y * newPos.y < 0.0) {
      float t = pos.y / (pos.y - newPos.y);
      vec3 hit = mix(pos, newPos, t);
      accum += sampleDisk(hit, newDir);
    }

    pos = newPos;
    dir = newDir;
  }

  if (escaped) accum += starfield(dir);

  vec3 col = accum / (accum + vec3(1.0));
  col = pow(col, vec3(1.0 / 1.85));
  gl_FragColor = vec4(col, 1.0);
}
`;

interface BlackHoleCanvasProps {
  className?: string;
  /** Internal render scale; 0.5 halves both axes before the browser upsizes. */
  quality?: number;
}

export function BlackHoleCanvas({
  className,
  quality = 0.5,
}: BlackHoleCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl", {
      antialias: false,
      powerPreference: "high-performance",
    }) as WebGLRenderingContext | null;
    if (!gl) {
      // Leave the canvas blank — the surrounding gradient keeps the layout
      // from collapsing, and we don't want to force-error the page.
      return;
    }

    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)")
      .matches;

    // ---- program -------------------------------------------------------
    const program = linkProgram(gl, VERT_SRC, FRAG_SRC);
    if (!program) return;

    const aPos = gl.getAttribLocation(program, "aPos");
    const uResolution = gl.getUniformLocation(program, "uResolution");
    const uTime = gl.getUniformLocation(program, "uTime");
    const uCamPos = gl.getUniformLocation(program, "uCamPos");
    const uCamRot = gl.getUniformLocation(program, "uCamRot");

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );

    gl.useProgram(program);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    // ---- camera state --------------------------------------------------
    let theta = 0.7;
    let phi = Math.PI / 2 - 0.16;
    let radius = 13;

    let dragging = false;
    let lx = 0;
    let ly = 0;
    let pinchStart = 0;
    let touchPrev: { x: number; y: number } | null = null;
    let lastInteraction = performance.now();

    function resize() {
      const rect = canvas!.getBoundingClientRect();
      const w = Math.max(1, Math.floor(rect.width * quality));
      const h = Math.max(1, Math.floor(rect.height * quality));
      if (canvas!.width !== w || canvas!.height !== h) {
        canvas!.width = w;
        canvas!.height = h;
      }
      gl!.viewport(0, 0, w, h);
      gl!.uniform2f(uResolution, w, h);
    }

    function onMouseDown(e: MouseEvent) {
      dragging = true;
      lx = e.clientX;
      ly = e.clientY;
    }
    function onMouseUp() {
      dragging = false;
    }
    function onMouseMove(e: MouseEvent) {
      if (!dragging) return;
      theta -= (e.clientX - lx) * 0.005;
      phi -= (e.clientY - ly) * 0.005;
      phi = Math.max(0.08, Math.min(Math.PI - 0.08, phi));
      lx = e.clientX;
      ly = e.clientY;
      lastInteraction = performance.now();
    }
    function onWheel(e: WheelEvent) {
      e.preventDefault();
      radius *= 1 + e.deltaY * 0.001;
      radius = Math.max(3.5, Math.min(45, radius));
      lastInteraction = performance.now();
    }
    function onTouchStart(e: TouchEvent) {
      if (e.touches.length === 1) {
        touchPrev = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        pinchStart = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY,
        );
      }
      lastInteraction = performance.now();
    }
    function onTouchMove(e: TouchEvent) {
      e.preventDefault();
      if (e.touches.length === 1 && touchPrev) {
        theta -= (e.touches[0].clientX - touchPrev.x) * 0.005;
        phi -= (e.touches[0].clientY - touchPrev.y) * 0.005;
        phi = Math.max(0.08, Math.min(Math.PI - 0.08, phi));
        touchPrev = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        const d = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY,
        );
        if (pinchStart > 0) radius *= pinchStart / d;
        radius = Math.max(3.5, Math.min(45, radius));
        pinchStart = d;
      }
      lastInteraction = performance.now();
    }

    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mouseup", onMouseUp);
    window.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();

    // ---- render loop ---------------------------------------------------
    const start = performance.now();
    let raf = 0;
    let running = true;

    function onVisibility() {
      if (document.hidden) {
        running = false;
        cancelAnimationFrame(raf);
      } else if (!running) {
        running = true;
        lastInteraction = performance.now();
        raf = requestAnimationFrame(frame);
      }
    }
    document.addEventListener("visibilitychange", onVisibility);

    function frame() {
      if (!running) return;
      const now = performance.now();

      // Idle spin so the page stays alive if the user isn't interacting.
      // Reduced motion: freeze the camera entirely.
      if (!reduced && !dragging && now - lastInteraction > 2500) {
        theta += 0.0013;
      }

      const cx = radius * Math.sin(phi) * Math.cos(theta);
      const cy = radius * Math.cos(phi);
      const cz = radius * Math.sin(phi) * Math.sin(theta);
      gl!.uniform3f(uCamPos, cx, cy, cz);

      // Build look-at basis (right, up, -forward) targeting the origin.
      const fx = -cx,
        fy = -cy,
        fz = -cz;
      const fl = Math.hypot(fx, fy, fz) || 1;
      const fwX = fx / fl,
        fwY = fy / fl,
        fwZ = fz / fl;
      // right = fwd x worldUp(0,1,0)
      let rx = fwY * 0 - fwZ * 1;
      let ry = fwZ * 0 - fwX * 0;
      let rz = fwX * 1 - fwY * 0;
      const rl = Math.hypot(rx, ry, rz) || 1;
      rx /= rl;
      ry /= rl;
      rz /= rl;
      // up = right x fwd
      const uxv = ry * fwZ - rz * fwY;
      const uyv = rz * fwX - rx * fwZ;
      const uzv = rx * fwY - ry * fwX;

      // Column-major 3x3: columns are right, up, -forward.
      gl!.uniformMatrix3fv(
        uCamRot,
        false,
        new Float32Array([
          rx, ry, rz,
          uxv, uyv, uzv,
          -fwX, -fwY, -fwZ,
        ]),
      );

      gl!.uniform1f(uTime, reduced ? 0 : (now - start) / 1000);
      gl!.drawArrays(gl!.TRIANGLES, 0, 6);

      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      document.removeEventListener("visibilitychange", onVisibility);
      canvas.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mouseup", onMouseUp);
      window.removeEventListener("mousemove", onMouseMove);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      ro.disconnect();
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
      const lose = gl.getExtension("WEBGL_lose_context");
      lose?.loseContext();
    };
  }, [quality]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className={className}
      style={{ display: "block", width: "100%", height: "100%" }}
    />
  );
}

function linkProgram(
  gl: WebGLRenderingContext,
  vertSrc: string,
  fragSrc: string,
): WebGLProgram | null {
  const vert = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  if (!vert || !frag) return null;
  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("black hole shader link:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  src: string,
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("black hole shader compile:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}
