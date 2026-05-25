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

cleanup_preflight() {
  docker rm -f ai-fullstack-course-preflight >/dev/null 2>&1 || true
}

cleanup_docker_space() {
  echo "📦 Docker 磁盘占用（清理前）:"
  docker system df || true
  echo "🧹 构建前清理停止容器、未使用镜像和构建缓存..."
  cleanup_preflight
  docker container prune -f || true
  docker image prune -a -f || true
  docker builder prune -a -f || true
  echo "📦 Docker 磁盘占用（清理后）:"
  docker system df || true
}

trap cleanup_preflight EXIT

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
echo "🧹 清理本地构建产物..."
rm -rf node_modules dist .astro
cleanup_docker_space
echo "🔨 构建 Docker 镜像（构建期间旧容器仍在服务）..."
$COMPOSE build ai-course

# 3. 预热验证新镜像。验证失败时不替换线上容器，旧容器继续服务
echo "🧪 预热验证新镜像..."
cleanup_preflight
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
  cleanup_preflight
  exit 1
fi

check_preflight_html() {
  path="$1"
  expected="$2"
  docker run --rm --network proxy-net curlimages/curl:latest -sf --connect-timeout 5 "http://ai-fullstack-course-preflight:3000${path}" | grep -Fq "$expected"
}

check_preflight_redirect() {
  path="$1"
  expected_location="$2"
  docker run --rm --network proxy-net curlimages/curl:latest -sI --connect-timeout 5 "http://ai-fullstack-course-preflight:3000${path}" | grep -Fiq "location: ${expected_location}"
}

check_preflight_status() {
  path="$1"
  expected_status="$2"
  status="$(docker run --rm --network proxy-net curlimages/curl:latest -s -o /dev/null -w '%{http_code}' --connect-timeout 5 "http://ai-fullstack-course-preflight:3000${path}")"
  [ "$status" = "$expected_status" ]
}

echo "🌐 检查多语言构建产物..."
if ! check_preflight_html "/" 'lang="en-US"'; then
  echo "❌ 根路径不是英文默认语言，停止替换线上容器"
  docker logs --tail=50 ai-fullstack-course-preflight || true
  cleanup_preflight
  exit 1
fi
if ! check_preflight_html "/" 'value="/zh-cn/"' || ! check_preflight_html "/" 'value="/ja/"'; then
  echo "❌ 根路径语言切换链接缺失，停止替换线上容器"
  docker logs --tail=50 ai-fullstack-course-preflight || true
  cleanup_preflight
  exit 1
fi
if ! check_preflight_html "/zh-cn/" 'lang="zh-CN"'; then
  echo "❌ 中文路径构建异常，停止替换线上容器"
  docker logs --tail=50 ai-fullstack-course-preflight || true
  cleanup_preflight
  exit 1
fi
if ! check_preflight_html "/ja/" 'lang="ja-JP"'; then
  echo "❌ 日文路径构建异常，停止替换线上容器"
  docker logs --tail=50 ai-fullstack-course-preflight || true
  cleanup_preflight
  exit 1
fi
if ! check_preflight_redirect "/zh-Hans/" "/zh-cn/"; then
  echo "❌ 旧中文路径兼容跳转异常，停止替换线上容器"
  docker logs --tail=50 ai-fullstack-course-preflight || true
  cleanup_preflight
  exit 1
fi
if ! check_preflight_status "/not-real-page-for-status-test" "404"; then
  echo "❌ 缺失页面没有返回 404，停止替换线上容器"
  docker logs --tail=50 ai-fullstack-course-preflight || true
  cleanup_preflight
  exit 1
fi
echo "✅ 多语言构建检查通过"
cleanup_preflight

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
