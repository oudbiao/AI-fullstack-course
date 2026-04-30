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

if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
else
  COMPOSE="docker-compose"
fi

# 1. 拉取最新代码
echo "📥 拉取最新代码..."
if [ ! -d .git ]; then
  git init
  git remote add origin https://github.com/oudbiao/AI-fullstack-course.git
fi

git fetch origin
git checkout master 2>/dev/null || git checkout main 2>/dev/null || true
git pull origin master 2>/dev/null || git pull origin main 2>/dev/null || true

# 2. 先构建新镜像，旧容器继续服务
echo "🔨 构建 Docker 镜像（构建期间旧容器仍在服务）..."
$COMPOSE build ai-course

# 3. 预热验证新镜像。验证失败时不替换线上容器，旧容器继续服务
echo "🧪 预热验证新镜像..."
docker rm -f ai-fullstack-course-preflight >/dev/null 2>&1 || true
docker run -d --name ai-fullstack-course-preflight --network proxy-net ai-fullstack-course:latest
PREFLIGHT_READY=0
for i in $(seq 1 18); do
  if docker run --rm --network proxy-net curlimages/curl:latest -sf --connect-timeout 5 http://ai-fullstack-course-preflight:3000/ >/dev/null 2>&1; then
    echo "✅ 新镜像预热通过"
    PREFLIGHT_READY=1
    break
  fi
  sleep 5
done
if [ "$PREFLIGHT_READY" -eq 0 ]; then
  echo "❌ 新镜像预热失败，保留旧容器继续服务"
  docker logs --tail=50 ai-fullstack-course-preflight || true
  docker rm -f ai-fullstack-course-preflight >/dev/null 2>&1 || true
  exit 1
fi
docker rm -f ai-fullstack-course-preflight >/dev/null 2>&1 || true

# 4. 构建和预热都成功后再替换容器，降低线上不可用时间
echo "🔁 替换线上容器..."
$COMPOSE up -d --no-deps --force-recreate ai-course

# 5. 等待应用就绪
echo "⏳ 等待应用启动..."
sleep 10

# 6. 检查容器状态
echo "📊 检查容器状态..."
$COMPOSE ps

# 7. 清理旧镜像
echo "🧹 清理旧镜像..."
docker image prune -f

echo ""
echo "========================================="
echo "✅ 部署完成！"
echo "========================================="
echo "🌐 访问地址: http://localhost:3000"
echo "📊 容器状态:"
$COMPOSE ps
echo "📋 查看日志: $COMPOSE logs -f ai-course"
