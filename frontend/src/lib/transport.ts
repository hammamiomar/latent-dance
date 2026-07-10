/**
 * transport — module-level handle on the streaming WebSocket.
 *
 * The socket's LIFECYCLE (connect, reconnect, demux) is owned by
 * useWebSocket; this module only holds a reference to the active socket so
 * that stores, control functions, and utilities can send without receiving
 * callbacks through props. send() reads the reference at call time — the
 * exact `ws.current` semantics the hook wrappers had.
 *
 * LEAF MODULE: imports nothing from stores or hooks. Everything imports this.
 */

let socket: WebSocket | null = import.meta.hot?.data?.socket ?? null;

// Survive HMR of this module without dropping the live connection.
if (import.meta.hot) {
  import.meta.hot.dispose((data) => {
    data.socket = socket;
  });
}

export function attachSocket(ws: WebSocket): void {
  socket = ws;
}

/** Detach only if `ws` is still active — a stale close event must not clobber a newer connection. */
export function detachSocket(ws: WebSocket): void {
  if (socket === ws) socket = null;
}

/** Send one JSON control message; silently dropped when not connected. */
export function send(payload: Record<string, unknown>): void {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  }
}

/** Test-only: forget the attached socket. */
export function __resetTransport(): void {
  socket = null;
}
