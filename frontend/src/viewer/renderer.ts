import { colormapLut } from "./colormaps";

/** Pan/zoom state. ``zoom`` is CSS pixels per data pixel; (cx, cy) is the data
 * coordinate (in full-resolution pixels) at the centre of the canvas. */
export interface ViewTransform {
  zoom: number;
  cx: number;
  cy: number;
}

export interface DisplayParams {
  lo: number;
  hi: number;
  gamma: number;
  colormap: string;
}

const VERT = `#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }`;

// Every fragment maps back to data coordinates. When zoomed out, up to 4 × 4
// texels under the fragment are averaged (texelFetch: float textures don't
// need to be filterable), which avoids aliasing on atomic-resolution images.
const FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D u_data;
uniform sampler2D u_lut;
uniform vec2 u_canvas;
uniform vec2 u_center;
uniform float u_zoom;
uniform float u_bin;
uniform ivec2 u_texSize;
uniform float u_lo;
uniform float u_hi;
uniform float u_gamma;
uniform vec4 u_bg;
out vec4 outColor;

void main() {
  vec2 frag = vec2(gl_FragCoord.x, u_canvas.y - gl_FragCoord.y);
  vec2 d = (frag - 0.5 * u_canvas) / u_zoom + u_center;
  vec2 size = vec2(u_texSize);
  vec2 t = d / u_bin;
  if (t.x < 0.0 || t.y < 0.0 || t.x >= size.x || t.y >= size.y) { outColor = u_bg; return; }

  float fp = 1.0 / (u_zoom * u_bin);
  int n = int(clamp(ceil(fp), 1.0, 4.0));
  float sum = 0.0;
  float cnt = 0.0;
  for (int i = 0; i < 4; i++) {
    if (i >= n) break;
    for (int j = 0; j < 4; j++) {
      if (j >= n) break;
      vec2 s = t + ((vec2(float(i), float(j)) + 0.5) / float(n) - 0.5) * fp;
      ivec2 p = clamp(ivec2(floor(s)), ivec2(0), u_texSize - 1);
      float v = texelFetch(u_data, p, 0).r;
      if (!isnan(v) && !isinf(v)) { sum += v; cnt += 1.0; }
    }
  }
  if (cnt == 0.0) { outColor = u_bg; return; }
  float x = clamp((sum / cnt - u_lo) / max(u_hi - u_lo, 1e-30), 0.0, 1.0);
  x = pow(x, u_gamma);
  outColor = texture(u_lut, vec2(x * (255.0 / 256.0) + 0.5 / 256.0, 0.5));
}`;

const UNIFORMS = [
  "u_data", "u_lut", "u_canvas", "u_center", "u_zoom", "u_bin",
  "u_texSize", "u_lo", "u_hi", "u_gamma", "u_bg",
] as const;

export const VIEWPORT_BG = [0.06, 0.067, 0.082, 1] as const;

/** Draws a float32 image with a colormap on a WebGL2 canvas. */
export class ImageRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private dataTex: WebGLTexture;
  private lutTex: WebGLTexture;
  private loc = {} as Record<(typeof UNIFORMS)[number], WebGLUniformLocation | null>;
  private texSize: [number, number] = [0, 0];
  private bin = 1;
  private lutName = "";

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    this.gl = gl;
    this.program = link(gl, VERT, FRAG);
    for (const name of UNIFORMS) this.loc[name] = gl.getUniformLocation(this.program, name);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const pos = gl.getAttribLocation(this.program, "a_pos");
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);

    this.dataTex = makeTexture(gl, gl.NEAREST);
    this.lutTex = makeTexture(gl, gl.LINEAR);
  }

  setData(data: Float32Array, width: number, height: number, bin: number) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.dataTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);
    this.texSize = [width, height];
    this.bin = bin;
  }

  clearData() {
    this.texSize = [0, 0];
  }

  private setColormap(name: string) {
    if (name === this.lutName) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, colormapLut(name));
    this.lutName = name;
  }

  draw(view: ViewTransform, display: DisplayParams, dpr: number) {
    const { gl, canvas, loc } = this;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(...VIEWPORT_BG);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.texSize[0]) return;

    this.setColormap(display.colormap);
    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.dataTex);
    gl.uniform1i(loc.u_data, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
    gl.uniform1i(loc.u_lut, 1);

    gl.uniform2f(loc.u_canvas, canvas.width, canvas.height);
    gl.uniform2f(loc.u_center, view.cx, view.cy);
    gl.uniform1f(loc.u_zoom, view.zoom * dpr);
    gl.uniform1f(loc.u_bin, this.bin);
    gl.uniform2i(loc.u_texSize, this.texSize[0], this.texSize[1]);
    gl.uniform1f(loc.u_lo, display.lo);
    gl.uniform1f(loc.u_hi, display.hi);
    gl.uniform1f(loc.u_gamma, display.gamma);
    gl.uniform4f(loc.u_bg, ...VIEWPORT_BG);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  dispose() {
    const gl = this.gl;
    gl.deleteTexture(this.dataTex);
    gl.deleteTexture(this.lutTex);
    gl.deleteProgram(this.program);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}

function makeTexture(gl: WebGL2RenderingContext, filter: number): WebGLTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const compile = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  return p;
}
