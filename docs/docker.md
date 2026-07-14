> 本文为 docs/docker.md 的中文翻译，仅供阅读参考，以英文原文为准。

### Docker 配置（Docker Setup）

本仓库中的所有示例都提供了正常运行和使用 Docker 运行两种方式的说明。虽然并非必需，但推荐使用 Docker 方式，因为这会简化软件安装、产生更稳定的环境，并且对于依赖 ROS 的示例，还能让你避免安装 ROS 而弄乱你的机器。

- 基本的 Docker 安装说明见[此处](https://docs.docker.com/engine/install/)。
- Docker 必须以 [rootless 模式（rootless mode）](https://docs.docker.com/engine/security/rootless/)安装。
- 要使用你的 GPU，你还必须安装 [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
- 通过 `snap` 安装的 docker 版本与 NVIDIA container toolkit 不兼容，会导致其无法访问 `libnvidia-ml.so`（[issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)）。可以用 `sudo snap remove docker` 卸载 snap 版本。
- Docker Desktop 同样与 NVIDIA runtime 不兼容（[issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)）。可以用 `sudo apt remove docker-desktop` 卸载 Docker Desktop。


如果你从零开始，且宿主机（host machine）是 Ubuntu 22.04，你可以使用便捷脚本 `scripts/docker/install_docker_ubuntu22.sh` 和 `scripts/docker/install_nvidia_container_toolkit.sh` 完成上述所有操作。

用以下命令构建 Docker 镜像并启动容器：
```bash
docker compose -f scripts/docker/compose.yml up --build
```

要为特定示例构建并运行 Docker 镜像，使用以下命令：
```bash
docker compose -f examples/<example_name>/compose.yml up --build
```
其中 `<example_name>` 是你想运行的示例的名称。

在首次运行任何示例时，Docker 会构建镜像。这期间你可以去喝杯咖啡。由于镜像会被缓存，后续运行会更快。
