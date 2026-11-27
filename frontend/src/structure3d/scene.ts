import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";
import type { DrawnAtoms, ElementInfo, Rgb, Vec3 } from "./structure";

// The first view: oblique, so the cell reads as 3D.
const START_DIRECTION = new THREE.Vector3(0.9, -1.7, 0.7).normalize();
const BEAM_DIRECTION = new THREE.Vector3(0, 0, 1);
const MARGIN = 1.12;
// A cell axis this aligned with the view (|cos|) can't be its up direction.
const UP_LIMIT = 0.98;
const BOND_RADIUS = 0.11;
const VACANCY_COLOR = 0xd8dce3;
// Coarser spheres and bonds for more atoms: [atoms up to, sphere segments, rings, bond sides].
type Detail = [number, number, number, number];
const DETAIL: Detail[] = [[2000, 32, 20, 12], [12000, 16, 10, 8], [Infinity, 10, 6, 5]];
const EXPORT_LONG_SIDE = 2400;
const MAX_EXPORT_RATIO = 4;

const colorOf = ([r, g, b]: Rgb) => new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);

/** What the scene shows. ``scale`` is × the covalent radius, ``hidden`` the
 * elements left out. */
export interface SceneModel {
  atoms: DrawnAtoms;
  bonds: [number, number][];
  sites: Record<string, number>[];
  elements: Record<string, ElementInfo>;
  repeats: number[];
  scale: number;
  hidden: string[];
}

export interface SceneOptions {
  onHover: (index: number | null) => void;
  onAxes: (a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3) => void;
  onView: (direction: number[], up: number[]) => void;
  beamFirst?: boolean;
}

export interface Scene {
  setModel(next: SceneModel): void;
  setCellVisible(visible: boolean): void;
  reset(): void;
  lookAlong(direction: number[]): void;
  /** Look down ``view`` (Cartesian, towards the viewer) with ``up`` up the screen. */
  setOrientation(view: number[], up: number[]): void;
  tilt(right: number, up: number): void;
  roll(degrees: number): void;
  toPng(): string;
  dispose(): void;
}

type Meshes = THREE.InstancedMesh<THREE.BufferGeometry, THREE.Material>;

/** An orthographic view of a structure with the unit cell ``cell`` in
 * ``container``: left-drag rotates, wheel zooms, right-drag pans. What it
 * shows is given to ``setModel``. ``onHover(index | null)`` names the atom
 * under the pointer; ``onAxes(a, b, c)`` gives the cell directions in view
 * coordinates; ``onView([x, y, z], up)`` the Cartesian direction to the viewer
 * and the screen's up direction (Cartesian). */
export function createScene(
  container: HTMLElement,
  cell: Vec3[],
  { onHover, onAxes, onView, beamFirst = false }: SceneOptions,
): Scene {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);
  const canvas = renderer.domElement;

  const [a, b, c] = cell.map((v) => new THREE.Vector3(...v));
  const center = new THREE.Vector3();
  let radius = 1;
  let beamZoom = 1;
  let model: SceneModel | null = null;
  let atomRadii = new Float32Array(0);

  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
  scene.add(camera);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x555566, 1.5));
  const light = new THREE.DirectionalLight(0xffffff, 1.8);
  light.position.set(-1, 1.5, 2);
  camera.add(light);
  camera.add(light.target);
  light.target.position.set(0, 0, -1);

  const atomGroup = new THREE.Group();
  const bondGroup = new THREE.Group();
  scene.add(atomGroup, bondGroup);

  const cellLines = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x8b93a1 }),
  );
  scene.add(cellLines);

  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(1, 24, 16),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35, depthWrite: false }),
  );
  halo.visible = false;
  scene.add(halo);

  const controls = new TrackballControls(camera, canvas);
  controls.rotateSpeed = 3;
  controls.zoomSpeed = 5;
  controls.panSpeed = 1;
  controls.staticMoving = true;
  controls.keys = ["", "", ""]; // the app has its own keys

  let dirty = true;
  controls.addEventListener("change", () => {
    dirty = true;
  });

  const fit = () => {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    renderer.setSize(width, height);
    const aspect = width / height;
    const half = radius * MARGIN * (aspect < 1 ? 1 / aspect : 1);
    camera.left = -half * aspect;
    camera.right = half * aspect;
    camera.top = half;
    camera.bottom = -half;
    camera.far = radius * 20;
    camera.updateProjectionMatrix();
    controls.handleResize();
    dirty = true;
  };

  const reset = () => {
    // A model starts looking down the beam (z), with y up.
    camera.position.copy(center).addScaledVector(beamFirst ? BEAM_DIRECTION : START_DIRECTION, radius * 6);
    camera.up.set(0, beamFirst ? 1 : 0, beamFirst ? 0 : 1);
    camera.zoom = beamFirst ? beamZoom : 1;
    controls.target.copy(center);
    camera.lookAt(center);
    camera.updateProjectionMatrix();
    dirty = true;
  };

  const observer = new ResizeObserver(fit);
  observer.observe(container);

  // Hovering: looked up once a frame, for the last pointer position.
  const raycaster = new THREE.Raycaster();
  let pointer: THREE.Vector2 | null = null;
  let hovered: number | null = null;
  const setHovered = (atom: number | null) => {
    if (atom === hovered) return;
    hovered = atom;
    halo.visible = atom !== null;
    if (atom !== null && model) {
      halo.position.set(...model.atoms.xyz[atom]);
      halo.scale.setScalar(atomRadii[atom] * 1.2);
    }
    dirty = true;
    onHover(atom);
  };
  const onMove = (e: PointerEvent) => {
    if (e.buttons) {
      pointer = null;
      setHovered(null);
      return;
    }
    const r = canvas.getBoundingClientRect();
    pointer = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  };
  const onLeave = () => {
    pointer = null;
    setHovered(null);
  };
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);

  const inverse = new THREE.Quaternion();
  const reportAxes = () => {
    inverse.copy(camera.quaternion).invert();
    const [x, y, z] = [a, b, c].map((v) => v.clone().normalize().applyQuaternion(inverse));
    onAxes(x, y, z);
    onView(camera.position.clone().sub(controls.target).toArray(), camera.up.toArray());
  };

  /** Turn the view by ``q``, about the point it looks at. */
  const turn = (q: THREE.Quaternion) => {
    const eye = camera.position.clone().sub(controls.target).applyQuaternion(q);
    camera.position.copy(controls.target).add(eye);
    camera.up.applyQuaternion(q).normalize();
    camera.lookAt(controls.target);
    dirty = true;
  };
  const about = (axis: THREE.Vector3, degrees: number) =>
    new THREE.Quaternion().setFromAxisAngle(axis, THREE.MathUtils.degToRad(degrees));

  const clear = (group: THREE.Group) => {
    for (const child of [...group.children]) {
      const mesh = child as Meshes;
      group.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
      mesh.dispose();
    }
  };

  /** The atoms, one mesh per component of each site: a site with several
   * elements (or vacancies) is drawn as wedges in proportion. */
  const buildAtoms = ({ atoms, sites, elements, scale, hidden }: SceneModel, [, segments, rings]: Detail) => {
    clear(atomGroup);
    atomRadii = new Float32Array(atoms.xyz.length);
    const bySite = new Map<number, number[]>();
    atoms.site.forEach((s, i) => {
      const list = bySite.get(s);
      if (list) list.push(i);
      else bySite.set(s, [i]);
    });
    const matrix = new THREE.Matrix4();
    for (const [s, indices] of bySite) {
      const parts = Object.entries(sites[s]);
      const total = parts.reduce((sum, [, x]) => sum + x, 0);
      const r = (scale * parts.reduce((sum, [el, x]) => sum + elements[el].covalent * x, 0)) / total;
      indices.forEach((i) => {
        atomRadii[i] = r;
      });
      if (parts.every(([el]) => hidden.includes(el))) continue;
      const components: { el: string | null; x: number }[] = parts.map(([el, x]) => ({ el, x }));
      if (total < 1 - 1e-3) components.push({ el: null, x: 1 - total });
      const whole = components.length === 1;
      let start = 0;
      for (const { el, x } of components) {
        const length = (whole ? 1 : x / Math.max(total, 1)) * Math.PI * 2;
        if (el === null || !hidden.includes(el)) {
          const geometry = new THREE.SphereGeometry(r, segments, rings, start, length);
          if (!whole) geometry.rotateX(Math.PI / 2);
          const material = new THREE.MeshStandardMaterial({
            color: el === null ? VACANCY_COLOR : colorOf(elements[el].color),
            roughness: 0.45,
            metalness: 0.05,
            side: whole ? THREE.FrontSide : THREE.DoubleSide,
          });
          const mesh = new THREE.InstancedMesh(geometry, material, indices.length);
          indices.forEach((atom, k) => mesh.setMatrixAt(k, matrix.makeTranslation(...atoms.xyz[atom])));
          mesh.instanceMatrix.needsUpdate = true;
          mesh.computeBoundingSphere();
          mesh.userData.indices = indices;
          mesh.userData.wedge = !whole;
          atomGroup.add(mesh);
        }
        start += length;
      }
    }
  };

  /** Bonds between shown atoms, each half in its atom's colour. */
  const buildBonds = ({ atoms, bonds, sites, elements, hidden }: SceneModel, [, , , sides]: Detail) => {
    clear(bondGroup);
    const siteShown = sites.map((site) => Object.keys(site).some((el) => !hidden.includes(el)));
    const halves = new Map<string, [number, number][]>();
    for (const [i, j] of bonds) {
      if (!siteShown[atoms.site[i]] || !siteShown[atoms.site[j]]) continue;
      for (const [from, to] of [[i, j], [j, i]]) {
        const symbol = atoms.symbol[from];
        const list = halves.get(symbol);
        if (list) list.push([from, to]);
        else halves.set(symbol, [[from, to]]);
      }
    }
    const y = new THREE.Vector3(0, 1, 0);
    const p = new THREE.Vector3();
    const q = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const mid = new THREE.Vector3();
    const rotation = new THREE.Quaternion();
    const size = new THREE.Vector3();
    const matrix = new THREE.Matrix4();
    for (const [symbol, list] of halves) {
      const mesh = new THREE.InstancedMesh(
        new THREE.CylinderGeometry(BOND_RADIUS, BOND_RADIUS, 1, sides, 1, true),
        new THREE.MeshStandardMaterial({ color: colorOf(elements[symbol].color), roughness: 0.5 }),
        list.length,
      );
      list.forEach(([from, to], k) => {
        p.set(...atoms.xyz[from]);
        q.set(...atoms.xyz[to]);
        dir.subVectors(q, p);
        const length = dir.length() / 2;
        mid.copy(p).addScaledVector(dir, 0.25);
        rotation.setFromUnitVectors(y, dir.normalize());
        size.set(1, length, 1);
        mesh.setMatrixAt(k, matrix.compose(mid, rotation, size));
      });
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
      bondGroup.add(mesh);
    }
  };

  /** The outline of the repeated cells; sets what the view is centred on. */
  const frameRepeats = ([na, nb, nc]: number[], largest: number) => {
    const A = a.clone().multiplyScalar(na);
    const B = b.clone().multiplyScalar(nb);
    const C = c.clone().multiplyScalar(nc);
    center.copy(A).add(B).add(C).multiplyScalar(0.5);
    const corners = [0, 1, 2, 3, 4, 5, 6, 7].map((i) =>
      new THREE.Vector3().addScaledVector(A, i & 1).addScaledVector(B, (i >> 1) & 1).addScaledVector(C, (i >> 2) & 1),
    );
    radius = Math.max(...corners.map((p) => p.distanceTo(center))) + largest;
    // Down the beam, only the width across it has to fit.
    beamZoom = radius / (Math.max(A.length(), B.length()) / 2 + largest);
    const edges: THREE.Vector3[] = [];
    for (let i = 0; i < 8; i++) {
      for (const bit of [1, 2, 4]) if (!(i & bit)) edges.push(corners[i], corners[i | bit]);
    }
    cellLines.geometry.dispose();
    cellLines.geometry = new THREE.BufferGeometry().setFromPoints(edges);
  };

  // A mixed site's wedges face the viewer, so its proportions always show.
  const wedgeMatrix = new THREE.Matrix4();
  const wedgeAt = new THREE.Vector3();
  const wedgeScale = new THREE.Vector3(1, 1, 1);
  const faceWedges = () => {
    if (!model) return;
    const { atoms } = model;
    for (const child of atomGroup.children) {
      const mesh = child as THREE.InstancedMesh;
      if (!mesh.userData.wedge) continue;
      (mesh.userData.indices as number[]).forEach((atom, k) => {
        wedgeAt.set(...atoms.xyz[atom]);
        mesh.setMatrixAt(k, wedgeMatrix.compose(wedgeAt, camera.quaternion, wedgeScale));
      });
      mesh.instanceMatrix.needsUpdate = true;
    }
  };

  let frame = requestAnimationFrame(function tick() {
    controls.update();
    if (pointer && model) {
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(atomGroup.children, false)[0];
      setHovered(hit && hit.instanceId !== undefined ? hit.object.userData.indices[hit.instanceId] : null);
      pointer = null;
    }
    if (dirty && model) {
      dirty = false;
      faceWedges();
      renderer.render(scene, camera);
      reportAxes();
    }
    frame = requestAnimationFrame(tick);
  });

  return {
    /** Show ``next``. New repeats keep the viewing direction and fit the view
     * to them. */
    setModel(next) {
      const first = model === null;
      const repeated = first || next.repeats.join() !== model!.repeats.join();
      const eye = first ? null : camera.position.clone().sub(controls.target).normalize();
      setHovered(null);
      model = next;
      const detail = DETAIL.find(([limit]) => next.atoms.xyz.length <= limit)!;
      buildAtoms(next, detail);
      buildBonds(next, detail);
      if (repeated) {
        frameRepeats(next.repeats, atomRadii.reduce((largest, r) => Math.max(largest, r), 0));
        fit();
        if (first || !eye) reset();
        else {
          controls.target.copy(center);
          camera.position.copy(center).addScaledVector(eye, radius * 6);
          camera.zoom = 1;
          camera.lookAt(center);
          camera.updateProjectionMatrix();
        }
      }
      dirty = true;
    },
    setCellVisible(visible) {
      cellLines.visible = visible;
      dirty = true;
    },
    reset,
    /** Look down ``direction`` (Cartesian, towards the viewer), centred, at
     * the same zoom: c is up, or b when looking along c. */
    lookAlong(direction) {
      const d = new THREE.Vector3(...(direction as Vec3)).normalize();
      const reference = [c, b, a].find((v) => Math.abs(v.clone().normalize().dot(d)) < UP_LIMIT) ?? c;
      const up = reference.clone().addScaledVector(d, -reference.dot(d)).normalize();
      controls.target.copy(center);
      camera.position.copy(center).addScaledVector(d, radius * 6);
      camera.up.copy(up);
      camera.lookAt(center);
      dirty = true;
    },
    setOrientation(view, up) {
      const d = new THREE.Vector3(...(view as Vec3)).normalize();
      const u = new THREE.Vector3(...(up as Vec3));
      u.addScaledVector(d, -u.dot(d));
      if (u.lengthSq() < 1e-12) return this.lookAlong(view);
      controls.target.copy(center);
      camera.position.copy(center).addScaledVector(d, radius * 6);
      camera.up.copy(u.normalize());
      camera.lookAt(center);
      dirty = true;
    },
    /** Tilt the crystal on screen: to the right and up, in degrees. */
    tilt(right, up) {
      const q = new THREE.Quaternion();
      if (right) q.multiply(about(new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion), -right));
      if (up) q.multiply(about(new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion), up));
      turn(q);
    },
    /** Rotate the picture clockwise about the view direction, in degrees. */
    roll(degrees) {
      const axis = camera.position.clone().sub(controls.target).normalize();
      camera.up.applyQuaternion(about(axis, degrees)).normalize();
      camera.lookAt(controls.target);
      dirty = true;
    },
    /** The view as a PNG with a transparent background (a data URL). */
    toPng() {
      const width = Math.max(container.clientWidth, 1);
      const height = Math.max(container.clientHeight, 1);
      const ratio = Math.min(MAX_EXPORT_RATIO, EXPORT_LONG_SIDE / Math.max(width, height));
      const haloShown = halo.visible;
      halo.visible = false;
      renderer.setPixelRatio(ratio);
      renderer.setSize(width, height);
      faceWedges();
      renderer.render(scene, camera);
      const url = canvas.toDataURL("image/png");
      halo.visible = haloShown;
      renderer.setPixelRatio(window.devicePixelRatio);
      fit();
      return url;
    },
    dispose() {
      cancelAnimationFrame(frame);
      observer.disconnect();
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      controls.dispose();
      clear(atomGroup);
      clear(bondGroup);
      cellLines.geometry.dispose();
      cellLines.material.dispose();
      halo.geometry.dispose();
      halo.material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    },
  };
}
