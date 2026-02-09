# GitHub Actions 自动部署指南

## 工作流程

```
开发者 Push 代码到 GitHub
    ↓
GitHub Actions 自动触发
    ↓
通过 SSH 连接服务器
    ↓
执行部署脚本：git pull → docker-compose build → docker-compose up
    ↓
部署完成，自动更新网站
```

## 配置步骤

### 第 1 步：在服务器生成 SSH 密钥

```bash
# 生成不需要密码的 SSH 密钥
ssh-keygen -t rsa -b 4096 -f ~/.ssh/github_deploy -N ""

# 添加公钥到授权列表
cat ~/.ssh/github_deploy.pub >> ~/.ssh/authorized_keys

# 修改权限
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh

# 获取私钥内容（待会需要）
cat ~/.ssh/github_deploy
```

### 第 2 步：在 GitHub 仓库配置 Secrets

1. 进入 GitHub 仓库首页
2. Settings → Secrets and variables → Actions
3. 点击 "New repository secret"
4. 添加以下 3 个 Secrets：

#### Secret 1: SERVER_IP
- **Name**: `SERVER_IP`
- **Value**: 你的服务器 IP 地址（例如：`192.168.1.100`）

#### Secret 2: SERVER_USER
- **Name**: `SERVER_USER`
- **Value**: SSH 登录用户名（例如：`ubuntu`）

#### Secret 3: SERVER_SSH_KEY
- **Name**: `SERVER_SSH_KEY`
- **Value**: 第 1 步中 `~/.ssh/github_deploy` 的内容（整个私钥）

### 第 3 步：在服务器准备部署环境

```bash
# 创建部署目录
mkdir -p /opt/ai-course
cd /opt/ai-course

# 如果项目已存在，跳过 clone（Actions 会自动 git pull）
# git clone https://github.com/oudbiao/AI-fullstack-course.git .

# 确保 Docker 和 Docker Compose 已安装
docker --version
docker-compose --version

# 创建必要目录
mkdir -p logs
```

### 第 4 步：配置 docker-compose.yml

在项目根目录确保有 `docker-compose.yml`，内容类似：

```yaml
version: '3.8'

services:
  ai-course:
    build: .
    container_name: ai-fullstack-course
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    restart: unless-stopped
    networks:
      - ai-network

networks:
  ai-network:
    driver: bridge
```

### 第 5 步：配置 Dockerfile

项目根目录需要 `Dockerfile`：

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

## 测试部署

### 方法 1：通过 GitHub 界面测试

1. 进入仓库 → Actions 标签
2. 选择 "Auto Deploy to Server" 工作流
3. 点击 "Run workflow"
4. 选择分支（master 或 main）
5. 点击绿色 "Run workflow" 按钮

### 方法 2：通过推送代码测试

```bash
# 在本地做一个小改动
echo "# Test" >> README.md

# 提交并推送
git add .
git commit -m "test deploy"
git push origin master
```

然后到 GitHub 仓库的 Actions 标签查看部署进度。

## 查看部署日志

### 在 GitHub 中查看

1. 仓库 → Actions 标签
2. 选择最近的工作流运行
3. 点击 "deploy" 任务
4. 查看 "Deploy to Server via SSH" 步骤的日志

### 在服务器中查看

```bash
# Docker 容器日志
docker-compose logs -f ai-course

# 查看最近 100 行
docker-compose logs --tail=100 ai-course

# 查看特定时间段的日志
docker-compose logs --since 10m ai-course
```

## 故障排查

### 问题 1：Authentication failed

**症状**：部署时出现 SSH 认证错误

**解决**：
```bash
# 检查公钥是否正确添加到服务器
cat ~/.ssh/authorized_keys | grep $(cat ~/.ssh/github_deploy.pub)

# 重新添加公钥
cat ~/.ssh/github_deploy.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### 问题 2：Command timed out

**症状**：部署超时，可能是网络或构建时间过长

**解决**：
```bash
# 检查服务器网络连接
ping 8.8.8.8

# 手动运行部署脚本测试
cd /opt/ai-course
docker-compose down
docker-compose build  # 这可能需要几分钟
docker-compose up -d
```

### 问题 3：Docker build failed

**症状**：Docker 镜像构建失败

**解决**：
```bash
# 查看 Docker 日志
docker-compose build --no-cache

# 检查磁盘空间
df -h

# 清理旧镜像
docker system prune -a -f
```

### 问题 4：Port already in use

**症状**：容器启动失败，提示端口已被占用

**解决**：
```bash
# 查看占用 3000 端口的进程
lsof -i :3000

# 停止占用该端口的容器/进程
docker-compose down
ps aux | grep node
kill -9 <PID>
```

## 手动部署（不使用 GitHub Actions）

如果需要手动部署：

```bash
cd /opt/ai-course

# 1. 拉取最新代码
git pull origin master

# 2. 停止容器
docker-compose down

# 3. 构建镜像
docker-compose build

# 4. 启动容器
docker-compose up -d

# 5. 查看状态
docker-compose ps
```

## 高级配置

### 部署前运行测试

编辑 `.github/workflows/deploy.yml`，在 SSH 执行前添加：

```yaml
      - name: Run Tests
        run: |
          npm test
```

### 部署失败时发送通知

添加步骤：

```yaml
      - name: Send Slack Notification
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK }}
          payload: |
            {
              "text": "❌ 部署失败！",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "仓库: ${{ github.repository }}\nBranch: ${{ github.ref }}\nCommit: ${{ github.sha }}"
                  }
                }
              ]
            }
```

### 定时部署（每天凌晨2点）

修改 `on` 部分：

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # 每天 02:00 UTC
  push:
    branches:
      - master
      - main
```

## 安全建议

1. **定期轮换 SSH 密钥**
   ```bash
   # 每 6 个月生成新密钥
   ssh-keygen -t rsa -b 4096 -f ~/.ssh/github_deploy_new -N ""
   ```

2. **限制 SSH 访问权限**
   ```bash
   # 在 ~/.ssh/authorized_keys 中限制命令
   command="cd /opt/ai-course && bash scripts/deploy.sh" ssh-rsa AAAA...
   ```

3. **使用 IP 白名单**
   ```bash
   # 在防火墙中只允许 GitHub Actions 的 IP
   # GitHub IP 列表: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-githubs-ip-addresses
   ```

4. **监控部署日志**
   - 定期检查 GitHub Actions 日志
   - 设置部署失败通知
   - 监控服务器磁盘空间和内存

## 常用命令

```bash
# 查看工作流状态
# GitHub Actions 标签 → 选择工作流

# 重新运行失败的部署
# GitHub Actions → 选择失败的运行 → Re-run failed jobs

# 手动触发部署
# GitHub Actions → Auto Deploy to Server → Run workflow
```

## 获取帮助

- GitHub Actions 官方文档：https://docs.github.com/en/actions
- SSH 问题排查：运行 `ssh -v user@host` 查看详细日志
- Docker 问题：查看 `docker-compose logs`
