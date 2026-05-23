# Docker 部署指南

## 本地开发环境

### 当前推荐
- ✅ 本地内容开发优先使用 `npm run dev`
- ✅ 本地生产预览使用 `npm run build && npm run serve`
- ✅ Docker 用于接近生产环境的构建和 Nginx 静态服务验证

如果 Docker 无法拉取基础镜像，通常是 Docker daemon 网络或代理配置问题，而不是项目构建脚本问题。当前 Dockerfile 会清空构建容器内常见的宿主机代理变量，避免 `127.0.0.1` 代理地址被错误带进容器。

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

2. **直接构建**
```bash
docker compose build ai-course
```

3. **启动容器**
```bash
docker compose up -d ai-course
```

### Dockerfile 配置说明

Dockerfile 使用多阶段构建：
- `node:22-alpine` 阶段安装依赖并运行 `npm run build:docker`
- `nginx:1.27-alpine` 阶段只复制 `dist/` 静态产物
- 构建阶段设置低内存 `NODE_OPTIONS`，适配小型服务器
- 构建容器会清空常见代理环境变量，运行时镜像不包含本地代理设置

**本地 Docker 构建：**
```bash
docker compose build ai-course
```

**生产构建：**
```bash
docker compose build ai-course
```

## 最佳实践

### 开发阶段
- ✅ 使用 `npm run dev` 本地开发
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
4. 服务器上执行：`git fetch/reset → docker compose build → 预热验证 → docker compose up -d`
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
| 本地开发 | `npm run dev` | Astro 开发服务器 |
| 本地生产预览 | `npm run build && npm run serve` | 构建并预览静态站点 |
| 本地测试 Docker | `docker compose build ai-course` | 使用生产 Dockerfile 构建 |
| 生产构建 | `docker compose build ai-course` | 构建 Nginx 静态镜像 |
| 生产运行 | `docker compose up -d ai-course` | 后台运行 |
| 推送更新 | `git push origin main` | 触发 GitHub Actions |
