"""WebSocket client used by the Hermes MCP server.

Hermes runs locally and acts on the frontend/local desktop bridge. It should
not talk directly to the remote generation backend.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import uuid4

from hambajuba2ba.integrations.hermes.contracts import AgentApplyRequest


LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
BRIDGE_TYPES = frozenset(
    {
        "agent.get_state",
        "agent.get_control_surface",
        "agent.get_music_window",
        "agent.get_song_analysis",
        "agent.report_phase",
        "agent.apply_visual_plan",
    }
)


class HambaBridgeClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:14321",
        token: str | None = None,
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._validate_local_url()
        self._ws: Any | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def get_state(self) -> dict[str, Any]:
        return await self._request("agent.get_state")

    async def get_control_surface(self) -> dict[str, Any]:
        return await self._request("agent.get_control_surface")

    async def get_music_window(
        self,
        *,
        lookback: float = 8.0,
        lookahead: float = 16.0,
    ) -> dict[str, Any]:
        return await self._request(
            "agent.get_music_window",
            {"lookback": lookback, "lookahead": lookahead},
        )

    async def get_song_analysis(self) -> dict[str, Any]:
        return await self._request("agent.get_song_analysis")

    async def report_phase(self, event: dict[str, Any]) -> dict[str, Any]:
        return await self._request("agent.report_phase", event)

    async def apply_visual_plan(
        self,
        *,
        actions: list[dict[str, Any]],
        based_on_audio_time: float | None = None,
        based_on_wall_time_ms: int | None = None,
        max_staleness_sec: float | None = None,
        transcript: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        reason: str | None = None,
        feature_candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        plan = AgentApplyRequest.model_validate(
            {
                # Live Hermes steering is a durable rig mutation, not a
                # beat-perfect cue. Timing fields are accepted at the MCP API
                # for older agents, but intentionally not forwarded to the
                # frontend validator.
                "based_on_audio_time": None,
                "based_on_wall_time_ms": None,
                "max_staleness_sec": None,
                "actions": actions,
                "transcript": transcript,
                "provider": provider,
                "model": model,
                "reason": reason,
                "feature_candidates": feature_candidates or [],
            }
        )
        return await self._request(
            "agent.apply_visual_plan",
            plan.model_dump(mode="json", exclude_none=True),
        )

    async def _request(
        self,
        message_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if message_type not in BRIDGE_TYPES:
            raise ValueError(f"Refusing non-Hamba bridge message: {message_type}")

        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "Hermes integration requires websockets. Install with `uv sync --extra hermes`."
            ) from exc

        ws = await self._ensure_connected(websockets)
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future

        async with self._send_lock:
            await ws.send(
                json.dumps(
                    {
                        "id": request_id,
                        "type": message_type,
                        "payload": payload or {},
                    },
                    separators=(",", ":"),
                )
            )

        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        finally:
            self._pending.pop(request_id, None)

    async def _ensure_connected(self, websockets_module: Any) -> Any:
        async with self._connect_lock:
            if self._ws is not None:
                return self._ws

            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            try:
                self._ws = await websockets_module.connect(
                    self._websocket_url(),
                    additional_headers=headers or None,
                    open_timeout=self.timeout,
                )
            except TypeError:
                self._ws = await websockets_module.connect(
                    self._websocket_url(),
                    extra_headers=headers or None,
                    open_timeout=self.timeout,
                )
            self._reader_task = asyncio.create_task(self._read_loop())
            return self._ws

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                message = json.loads(raw)
                request_id = message.get("id")
                if not request_id:
                    continue
                future = self._pending.get(request_id)
                if future is None or future.done():
                    continue
                if message.get("type") == "error":
                    error = message.get("error") or {}
                    future.set_exception(
                        RuntimeError(error.get("message", "Bridge request failed"))
                    )
                else:
                    payload = message.get("payload")
                    future.set_result(
                        payload if isinstance(payload, dict) else {"value": payload}
                    )
        except Exception as exc:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(exc)
        finally:
            self._ws = None

    async def close(self) -> None:
        ws = self._ws
        reader_task = self._reader_task
        self._ws = None
        self._reader_task = None
        if ws is not None:
            await ws.close()
        if reader_task is not None:
            reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await reader_task

    def _validate_local_url(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https", "ws", "wss"}:
            raise ValueError(f"Unsupported Hamba bridge URL scheme: {parsed.scheme}")
        if parsed.hostname not in LOCAL_HOSTS:
            raise ValueError(
                "Hermes control must target the local frontend bridge, "
                f"not a remote host: {parsed.hostname}"
            )

    def _websocket_url(self) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
        path = parsed.path.rstrip("/")
        if not path or path == "/":
            path = "/agent/ws"
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        params["role"] = "mcp"
        query = urlencode(params)
        return urlunparse((scheme, parsed.netloc, path, "", query, ""))


def client_from_env() -> HambaBridgeClient:
    base_url = os.getenv("HAMBA_FRONTEND_BRIDGE_URL", "ws://127.0.0.1:14321/agent/ws")
    token = os.getenv("HAMBA_FRONTEND_BRIDGE_TOKEN") or os.getenv("HAMBA_CONTROL_TOKEN")
    return HambaBridgeClient(base_url=base_url, token=token)
