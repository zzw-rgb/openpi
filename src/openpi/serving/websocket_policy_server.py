"""WebSocket（双机长连接）推理服务：把 Policy 暴露成机器人端可远程调用的服务。

推理链路的部署方式是“双机分工”：策略模型跑在带 GPU 的推理机上，机器人本体端只负责采集观测、
执行动作，两者用 websocket 长连接双向通信。相比反复握手的一次性 HTTP，长连接更适合高频控制回路。
本文件是服务端。

关键类 WebsocketPolicyServer 包住一个 Policy：serve_forever/run 启动 asyncio 服务器（关压缩、
去掉单帧大小上限，以适配较大的图像/动作数组，并挂 /healthz 健康检查）；每个客户端连接对应一次
_handler 协程——建连先用 msgpack+numpy 编解码器把策略元数据发给客户端，随后进入主循环：
【收】反序列化客户端发来的观测帧 → 【算】调用 policy.infer 跑完整推理链路得到动作块 →
【发】把动作块（附服务端推理耗时）打包回传，如此一帧观测换一帧动作，直到连接关闭。异常时把完整
traceback 作为一帧发回客户端便于远程排错。

输入：客户端 websocket 送来的观测字典。输出：回传的动作字典（含 actions 与 timing）。
由 scripts/serve_policy.py 构造并 serve_forever 启动；对端客户端见 examples/droid/main.py
所用的 websocket_client_policy。
"""

import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


# 为什么用 websocket 远程推理：策略模型跑在带 GPU 的推理机上，机器人本体端只负责采集
# 观测、执行动作。两者用 websocket 长连接双向通信——机器人把观测打包发来，服务器推理后
# 把动作块回传。相比一次性 HTTP，长连接省去反复握手，适合高频控制回路。
class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    # 阻塞式启动：内部起一个 asyncio 事件循环长期服务。
    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        # 启动 websocket 服务器。compression=None、max_size=None：观测/动作是较大的数值数组，
        # 关压缩省 CPU、去掉单帧大小上限避免大图被拒；process_request 挂健康检查钩子。
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    # 每个客户端连接对应一次 _handler 协程：建连后先发元数据，然后进入“收观测→推理→回动作”循环。
    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        # msgpack + numpy 编解码器：把含 numpy 数组的字典高效序列化成二进制帧。
        packer = msgpack_numpy.Packer()

        # 握手：连接一建立就把策略元数据（如动作维度、默认 prompt 等）发给客户端。
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        # 主循环：一帧观测换一帧动作，直到连接关闭。
        while True:
            try:
                start_time = time.monotonic()
                # 【收】阻塞等待客户端发来的观测帧，反序列化回 numpy 字典。
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # 【算】调用 Policy.infer 跑完整推理链路，得到动作块。
                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                # 附带服务端耗时信息，便于客户端监控推理延迟。
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    # 上一轮“收+算+发”的总耗时（发的时间只能等下一轮才知道，故记上一轮的）。
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # 【发】把动作块打包回传给机器人端执行。
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                # 客户端正常断开：结束本连接的协程。
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                # 其他异常：把完整 traceback 作为一帧发回客户端便于远程排错，再带错误码关连接并抛出。
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


# 健康检查：对 /healthz 路径直接返回 200 OK（供负载均衡/存活探测），其余请求走正常 websocket 流程。
def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
