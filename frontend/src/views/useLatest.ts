import { useEffect, useRef, useState } from "react";
import { Coalescer } from "../api/coalesce";

/** The result of ``fetch`` for the latest ``key`` (null: nothing to fetch).
 * While a request is in flight newer keys replace each other, so dragging a
 * slider never queues stale requests. The last result stays shown until the
 * next arrives (``key`` says which it was for). */
export function useLatest<T>(key: string | null, fetch: () => Promise<T>) {
  const [state, setState] = useState<{ key: string; value: T } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const live = useRef(fetch);
  live.current = fetch;
  const coalescer = useRef<Coalescer<{ key: string; value: T }>>();
  coalescer.current ??= new Coalescer(
    (_c, v) => {
      setState(v);
      setError(null);
    },
    (_c, e) => setError(e.message),
  );

  useEffect(() => {
    if (key === null) return;
    const k = key;
    coalescer.current!.request("latest", k, async () => ({ key: k, value: await live.current() }));
  }, [key]);

  return { value: state?.value ?? null, key: state?.key ?? null, loading: key !== null && state?.key !== key, error };
}
