/**
 * Latest-only fetching per channel: while a request for a channel is in
 * flight, newer requests replace each other and only the most recent one is
 * sent when it lands. Dragging a cursor or region therefore never queues up
 * stale requests, and results always arrive in order.
 */
export class Coalescer<T> {
  private inFlight = new Set<string>();
  private wanted = new Map<string, { key: string; fetch: () => Promise<T> }>();
  private done = new Map<string, string>();

  constructor(
    private onResult: (channel: string, value: T) => void,
    private onError: (channel: string, error: Error) => void = () => {},
  ) {}

  /** Ask for ``key`` on ``channel``; nothing is fetched if it's already current. */
  request(channel: string, key: string, fetch: () => Promise<T>) {
    this.wanted.set(channel, { key, fetch });
    void this.pump(channel);
  }

  forget(channel: string) {
    this.wanted.delete(channel);
    this.done.delete(channel);
  }

  private async pump(channel: string) {
    if (this.inFlight.has(channel)) return;
    const w = this.wanted.get(channel);
    if (!w || this.done.get(channel) === w.key) return;
    this.inFlight.add(channel);
    try {
      const value = await w.fetch();
      this.done.set(channel, w.key);
      if (this.wanted.has(channel)) this.onResult(channel, value);
    } catch (e) {
      this.done.set(channel, w.key);
      this.onError(channel, e as Error);
    } finally {
      this.inFlight.delete(channel);
      const next = this.wanted.get(channel);
      if (next && next.key !== this.done.get(channel)) void this.pump(channel);
    }
  }
}
