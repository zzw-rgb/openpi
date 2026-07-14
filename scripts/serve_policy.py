"""起服务入口：命令行装配一个 Policy 并把它挂上 WebSocket 推理服务。

这是推理机（GPU 端）的启动脚本，把散落的组件串成一条可运行的服务：读命令行参数 → 装配 Policy →
（可选）包录制器 → 起 websocket 服务，阻塞等待机器人端连接。

关键结构：Args 是 tyro 自动映射的命令行参数（--env / --port / --default_prompt / --record /
--policy...）；EnvMode 枚举支持的平台（aloha、aloha_sim、droid、libero）；Checkpoint / Default
两种策略来源——显式给出“训练配置名 + checkpoint 目录”，或用 DEFAULT_CHECKPOINT 里各环境预设的
默认 checkpoint（多为 gs:// 远端，首次自动下载缓存）。create_policy/create_default_policy 调用
policy_config.create_trained_policy 完成装配；main 再取策略元数据、按需用 PolicyRecorder 包一层
落盘调试，最后用 host=0.0.0.0 监听所有网卡起 WebsocketPolicyServer 并 serve_forever。

输入：命令行参数。输出：一个持续运行、监听指定端口的推理服务进程。
与 policies/policy_config.py（装配）、serving/websocket_policy_server.py（服务）配合；
机器人端客户端如 examples/droid/main.py 连到本服务的 IP:port。
"""

import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# 各环境的默认 checkpoint（不显式指定 policy 时用）。dir 多为 gs:// 远端，首次会自动下载缓存。
# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


# 按命令行参数决定 Policy 来源：显式给了 Checkpoint 就从该 config+dir 装配；否则用环境默认。
def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            # 从指定训练配置名 + checkpoint 目录装配 Policy。
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


# 入口：装配 Policy → （可选）包一层录制器 → 起 websocket 服务，阻塞等待机器人端连接。
def main(args: Args) -> None:
    policy = create_policy(args)
    # 取策略元数据，握手时发给客户端。
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    # 开了 record 就用 PolicyRecorder 包一层，把每步输入输出落盘调试。
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # host=0.0.0.0 监听所有网卡，机器人端可用本机 IP + port 远程连接。
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    # 阻塞运行，持续服务推理请求。
    server.serve_forever()


if __name__ == "__main__":
    # tyro 把 Args 数据类自动映射成命令行参数（--env/--port/--policy.config 等）。
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
