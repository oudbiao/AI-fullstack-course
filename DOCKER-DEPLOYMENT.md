# Docker 部署指南

## 本地开发环境

### 当前状态
- ✅ 应用通过 `npm start` 运行在 `http://localhost:3000`
- ❌ Docker 构建因网络问题受阻（Docker daemon 无法连接到 Docker Hub）

### 原因
- 您的系统使用代理 (Surge) 监听 `127.0.0.1:6152`
- OrbStack (macOS Docker 替代品) 的 daemon 未能正确使用该代理
- Docker daemon 无法从 Docker Hub 拉取 `node:lts-bookworm` 镜像

## 生产环境部署

### 推荐方式：在生产服务器上构建

**优点：**
- 无需处理本地网络代理问题
- 生产环境镜像与实际运行环境一致
- 避免镜像依赖本地开发配置

**步骤：**

1. **在生产服务器上克隆代码**
```bash
cd /opt/ai-course
git clone https://github.com/oudbiao/AI-fullstack-course.git .
```

2. **无需代理参数，直接构建**
```bash
docker-compose build
```

3. **启动容器**
```bash
docker-compose up -d
```

### Dockerfile 配置说明

Dockerfile 已配置为：
- `ARG` 用于构建参数（仅在构建时有效）
- **生产环境中不包含代理设置**（干净的运行时环境）

**本地开发构建（需要代理）：**
```bash
docker-compose build \
  --build-arg http_proxy=http://127.0.0.1:6152 \
  --build-arg https_proxy=http://127.0.0.1:6152 \
  --build-arg all_proxy=socks5://127.0.0.1:6153
```

**生产构建（无需代理）：**
```bash
docker-compose build
```

## 最佳实践

### 开发阶段
- ✅ 使用 `npm start` 本地开发
- 优点：快速刷新、易于调试、无需 Docker

### 部署阶段
- ✅ 使用 GitHub Actions 自动部署
- ✅ 在生产服务器上构建 Docker 镜像
- ✅ 通过 SSH 触发部署（无需暴露端口）

## GitHub Actions 部署流程

见 [GITHUB-ACTIONS-DEPLOY.md](./GITHUB-ACTIONS-DEPLOY.md)

1. 代码推送到 GitHub
2. GitHub Actions 触发
3. 通过 SSH 连接到生产服务器
4. 服务器上执行：`git pull → docker-compose build → docker-compose up -d`
5. 应用自动更新和重启

## 故障排除

### Docker 本地构建失败

**症状：** `failed to resolve source metadata for docker.io/library/node`

**原因：** Docker daemon 无法连接到镜像源

**解决方案：**
1. 检查网络连接
2. 配置 Docker daemon 代理（`~/.docker/config.json`）
3. 重启 Docker/OrbStack
4. 或在网络无限制的服务器上构建

### 生产环境部署失败

**检查清单：**
- ✅ Git 仓库可访问
- ✅ Docker 和 Docker Compose 已安装
- ✅ 网络连接正常
- ✅ GitHub Actions Secrets 已配置
- ✅ SSH 密钥有效

## 快速参考

| 环节 | 命令 | 说明 |
|------|------|------|
| 本地开发 | `npm start` | 开发服务器，http://localhost:3000 |
| 本地测试Docker | `docker-compose build --build-arg ...` | 需要代理参数 |
| 生产构建 | `docker-compose build` | 无需代理参数 |
| 生产运行 | `docker-compose up -d` | 后台运行 |
| 推送更新 | `git push origin main` | 触发 GitHub Actions |

