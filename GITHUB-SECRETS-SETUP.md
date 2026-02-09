# GitHub Secrets 配置指南

## 问题
GitHub Actions 部署失败，提示：`error: missing server host`

## 原因
GitHub Secrets 未配置。部署工作流需要三个必需的密钥：

| Secret 名称 | 说明 | 示例 |
|-----------|------|------|
| `SERVER_IP` | 生产服务器 IP 地址 | `192.168.1.100` |
| `SERVER_USER` | SSH 登录用户名 | `ubuntu` 或 `root` |
| `SERVER_SSH_KEY` | SSH 私钥内容 | 见下方说明 |

## 配置步骤

### 1. 准备 SSH 密钥

**在生产服务器上生成 SSH 密钥对：**

```bash
# 生成 SSH 密钥（默认位置 ~/.ssh/id_rsa）
ssh-keygen -t rsa -b 4096 -C "ai-course-deploy" -N ""

# 查看公钥并添加到 authorized_keys
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 查看私钥内容（需要复制到 GitHub）
cat ~/.ssh/id_rsa
```

**或在本地生成后上传：**

```bash
# 本地生成
ssh-keygen -t rsa -b 4096 -C "ai-course-deploy" -N ""

# 显示私钥内容
cat ~/.ssh/id_rsa

# 上传公钥到服务器
ssh-copy-id -i ~/.ssh/id_rsa.pub username@server_ip
```

### 2. 在 GitHub 添加 Secrets

#### 方法 A：通过 Web 界面（推荐）

1. 打开 GitHub 仓库：https://github.com/oudbiao/AI-fullstack-course
2. 点击 **Settings** → **Secrets and variables** → **Actions**
3. 点击 **New repository secret** 按钮
4. 添加以下三个 Secrets：

**Secret 1: SERVER_IP**
- Name: `SERVER_IP`
- Value: `你的服务器 IP 地址` (例如: `192.168.1.100`)
- 点击 **Add secret**

**Secret 2: SERVER_USER**
- Name: `SERVER_USER`
- Value: `SSH 登录用户名` (例如: `ubuntu`)
- 点击 **Add secret**

**Secret 3: SERVER_SSH_KEY**
- Name: `SERVER_SSH_KEY`
- Value: `SSH 私钥内容`（完整内容，包括 `-----BEGIN RSA PRIVATE KEY-----` 和 `-----END RSA PRIVATE KEY-----`）
- 点击 **Add secret**

#### 方法 B：通过 GitHub CLI（高级）

```bash
# 需要先登录 GitHub CLI
gh auth login

# 进入仓库目录
cd /Users/carl/Downloads/ai-fullstack-course

# 添加 Secrets
gh secret set SERVER_IP --body "192.168.1.100"
gh secret set SERVER_USER --body "ubuntu"
gh secret set SERVER_SSH_KEY < ~/.ssh/id_rsa
```

### 3. 验证 Secrets 已添加

在 GitHub 仓库的 Settings → Secrets and variables → Actions 中应该看到：
- ✓ SERVER_IP
- ✓ SERVER_USER
- ✓ SERVER_SSH_KEY

### 4. 测试部署

Secrets 配置完成后，推送代码来触发自动部署：

```bash
cd /Users/carl/Downloads/ai-fullstack-course

# 确保所有更改已提交
git add .
git commit -m "Configure GitHub Actions deployment"

# 推送到 GitHub
git push origin master
```

## 监控部署进度

1. 打开 GitHub 仓库
2. 点击 **Actions** 标签
3. 查看 "Auto Deploy to Server" 工作流的执行状态
4. 点击具体的运行记录查看详细日志

## 部署流程

一旦 Secrets 配置正确，每次代码推送时会自动：

```
git push → GitHub Actions 触发
  ↓
通过 SSH 连接到服务器
  ↓
git pull 获取最新代码
  ↓
docker-compose build 构建镜像
  ↓
docker-compose up -d 启动容器
  ↓
✅ 部署完成
```

## 故障排除

### 错误：`Permission denied (publickey)`

**原因：** SSH 密钥无效或服务器上没有公钥

**解决方案：**
```bash
# 1. 检查服务器上的公钥
cat ~/.ssh/authorized_keys

# 2. 重新生成并添加公钥
ssh-keygen -t rsa -b 4096 -C "ai-course-deploy" -N ""
ssh-copy-id -i ~/.ssh/id_rsa.pub username@server_ip

# 3. 在 GitHub 更新 SERVER_SSH_KEY secret
```

### 错误：`missing server host`

**原因：** `SERVER_IP` secret 未设置

**解决方案：**
- 确保在 GitHub Settings → Secrets 中添加了 `SERVER_IP`
- 检查值不为空且格式正确

### 错误：`docker-compose: command not found`

**原因：** 生产服务器上未安装 Docker Compose

**解决方案：**
```bash
# 在生产服务器上安装
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

## 安全建议

- ✅ 使用强密码生成 SSH 密钥：`ssh-keygen -t rsa -b 4096`
- ✅ 使用专用部署账户（不要用 root）
- ✅ 限制 SSH 密钥权限：`chmod 600 ~/.ssh/id_rsa`
- ✅ 定期轮换 SSH 密钥
- ✅ GitHub Secrets 在日志中会被自动掩盖，不会泄露

## 相关文件

- 部署工作流：[.github/workflows/deploy.yml](.github/workflows/deploy.yml)
- 部署脚本：[scripts/deploy.sh](scripts/deploy.sh)
- Docker 配置：[docker-compose.yml](docker-compose.yml)
- 部署指南：[DOCKER-DEPLOYMENT.md](DOCKER-DEPLOYMENT.md)

