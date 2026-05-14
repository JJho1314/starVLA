# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import concurrent.futures
import logging
import time
import traceback

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # Idle timeout in seconds, -1 means never auto-close
        metadata: dict | None = None,
        batch_size: int = 1,        # 1 = no batching (per-request inference)
        batch_wait_ms: int = 20,    # max time to wait for more requests to fill a batch
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        self._batch_size = max(1, int(batch_size))
        self._batch_wait_s = max(0.0, float(batch_wait_ms) / 1000.0)
        # Single GPU executor; predict_action is sync/blocking and acquires CUDA context.
        self._infer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._infer_queue: asyncio.Queue | None = None  # created in run()
        self._batcher_task: asyncio.Task | None = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        # Create the request queue and start the batcher task in the same loop as the server.
        self._infer_queue = asyncio.Queue()
        if self._batch_size > 1:
            self._batcher_task = asyncio.create_task(self._batcher_loop())
            logging.info(
                f"Batched inference enabled: max_batch={self._batch_size}, wait={self._batch_wait_s*1000:.0f}ms"
            )
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            try:
                if self._idle_timeout > 0:
                    await self._idle_watchdog(server)
                else:
                    await server.serve_forever()
            finally:
                if self._batcher_task is not None:
                    self._batcher_task.cancel()

    async def _idle_watchdog(self, server):
        """Monitor idle time and shut down the server on timeout."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # Refresh active time on each received message
                ret = await self._route_message_async(msg)
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _route_message_async(self, msg: dict) -> dict:
        """Async wrapper around routing. For batch_size>1 the infer path is coalesced via the batcher;
        otherwise it falls back to the sync path (run in the executor so the loop stays free)."""
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")
        payload = msg.get("payload", msg)

        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        if mtype not in ("infer", "predict_action"):
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }

        if not isinstance(payload, dict):
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": req_id,
                "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
            }

        try:
            if self._batch_size > 1:
                # Each handler submits 1 example, awaits its slice of the batched result.
                fut: asyncio.Future = asyncio.get_running_loop().create_future()
                await self._infer_queue.put((payload, fut))
                output_dict = await fut
            else:
                # Run blocking inference in the executor to keep loop free for other connections
                loop = asyncio.get_running_loop()
                output_dict = await loop.run_in_executor(
                    self._infer_executor, lambda: self._policy.predict_action(**payload)
                )
        except Exception as e:
            logging.exception("Policy inference error (request_id=%s)", req_id)
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": req_id,
                "error": {"message": str(e)},
            }
        return {
            "status": "ok",
            "ok": True,
            "type": "inference_result",
            "request_id": req_id,
            "data": output_dict,
        }

    async def _batcher_loop(self):
        """Coalesce concurrent infer requests into a single predict_action(batch) call."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                first_payload, first_fut = await self._infer_queue.get()
            except asyncio.CancelledError:
                return
            items = [(first_payload, first_fut)]
            deadline = loop.time() + self._batch_wait_s
            # Drain queue up to max_batch within wait window
            while len(items) < self._batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self._infer_queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    return

            try:
                # Merge examples from all items; reuse first item's other kwargs (do_sample/use_ddim/...)
                merged_kwargs = dict(items[0][0])
                merged_kwargs.pop("examples", None)
                merged_examples = []
                slice_lens = []
                for payload, _ in items:
                    exs = payload.get("examples", [])
                    if not isinstance(exs, list):
                        exs = [exs]
                    merged_examples.extend(exs)
                    slice_lens.append(len(exs))
                merged_kwargs["examples"] = merged_examples

                output = await loop.run_in_executor(
                    self._infer_executor,
                    lambda: self._policy.predict_action(**merged_kwargs),
                )

                # Distribute slices back. We expect each value in `output` to be batch-major (np.ndarray
                # or list whose first axis matches the batch). Non-batched values get duplicated.
                offset = 0
                for (_, fut), n in zip(items, slice_lens):
                    sub: dict = {}
                    for k, v in output.items():
                        try:
                            if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] >= offset + n:
                                sub[k] = v[offset:offset + n]
                            elif isinstance(v, list) and len(v) >= offset + n:
                                sub[k] = v[offset:offset + n]
                            else:
                                sub[k] = v
                        except Exception:
                            sub[k] = v
                    if not fut.done():
                        fut.set_result(sub)
                    offset += n
            except Exception as e:
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(e)

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer
        payload = msg.get("payload", msg)  # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
                }
            try:
                output_dict = self._policy.predict_action(**payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)

                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                    },
                }
            data = output_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
