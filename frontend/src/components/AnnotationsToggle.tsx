import { useUi } from "../state/ui";

/** Shows or hides the annotations of image view ``view`` (regions, profile
 * lines, FFT boxes); hidden ones are kept. */
export function AnnotationsToggle({ view }: { view: string }) {
  const hidden = useUi((s) => !!s.annotationsHidden[view]);
  const setHidden = useUi((s) => s.setAnnotationsHidden);
  return (
    <button
      className={`icon-toggle${hidden ? " active" : ""}`}
      aria-pressed={hidden}
      onClick={() => setHidden(view, !hidden)}
      title={hidden ? "Show regions, lines and FFT boxes" : "Hide regions, lines and FFT boxes"}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden>
        <path d="M1.5 8C3 5 5.3 3.5 8 3.5S13 5 14.5 8C13 11 10.7 12.5 8 12.5S3 11 1.5 8Z" />
        <circle cx="8" cy="8" r="2" />
        {hidden && <path d="M2.5 13.5L13.5 2.5" />}
      </svg>
    </button>
  );
}
