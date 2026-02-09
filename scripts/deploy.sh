#!/bin/bash

# AI Fullstack Course - Docker 自动部署脚本
# 在服务器上手动执行或由 GitHub Actions 调用

set -e

echo "========================================="
echo "🚀 开始部署 AI Fullstack Course"
echo "========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "当前目录: $(pwd)"
echo ""

# 1. 拉取最新代码
echo "📥 拉取最新代码..."
if [ ! -d .git ]; then
  git init
  git remote add origin https://github.com/oudbiao/AI-fullstack-course.git
fi

git fetch origin
git checkout master 2>/dev/null || git checkout main 2>/dev/null || true
git pull origin master 2>/dev/null || git pull origin main 2>/dev/null || true

# 2. 停止旧容器
echo "🛑 停止旧容器..."
docker-compose down 2>/dev/null || true

# 3. 构建新镜像
echo "🔨 构建 Docker 镜像..."
docker-compose build --no-cache

# 4. 启动新容器
echo "▶️  启动新容器..."
docker-compose up -d

# 5. 等待应用就绪
echo "⏳ 等待应用启动..."
sleep 10

# 6. 检查容器状态
echo "📊 检查容器状态..."
docker-compose ps

# 7. 清理旧镜像
echo "🧹 清理旧镜像..."
docker image prune -f

echo ""
echo "========================================="
echo "✅ 部署完成！"
echo "========================================="
echo "🌐 访问地址: http://localhost:3000"
echo "📊 容器状态:"
docker-compose ps
echo "📋 查看日志: docker-compose logs -f ai-course"
