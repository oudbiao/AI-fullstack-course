FROM node:20-alpine AS builder

WORKDIR /app

# GitHub/server-side Docker builds can run on small machines.
# Keep Node below the host limit and render Docusaurus pages sequentially.
ENV NODE_OPTIONS="--max-old-space-size=1536"
ENV DOCUSAURUS_SSR_CONCURRENCY=1
ENV DOCUSAURUS_DISABLE_LAST_UPDATE=true
ENV CI=true
# Some local Docker runtimes inject host proxy variables that point at
# 127.0.0.1. Inside a build container that address is the container itself,
# so npm registry requests fail. Keep dependency install direct and explicit.
ENV HTTP_PROXY="" \
    HTTPS_PROXY="" \
    http_proxy="" \
    https_proxy="" \
    ALL_PROXY="" \
    all_proxy="" \
    npm_config_proxy="" \
    npm_config_https_proxy=""

# 先安装依赖，避免每次只改文档或图片都重新安装 node_modules
COPY package.json package-lock.json ./
RUN unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy npm_config_proxy npm_config_https_proxy \
    && npm config delete proxy || true \
    && npm config delete https-proxy || true \
    && npm install -g npm@11.6.2 --no-audit --no-fund \
    && npm ci --no-audit --no-fund \
    && test -x node_modules/.bin/docusaurus

# 再复制项目文件，最大化 Docker 构建缓存命中
COPY docusaurus.config.js ./
COPY sidebars.js ./
COPY scripts ./scripts
COPY docs ./docs
COPY i18n ./i18n
COPY src ./src
COPY static ./static

# 构建应用：Docker 使用低内存构建脚本，避免多语言站点在压缩阶段被 OOM kill。
RUN npm run build:docker

FROM nginx:1.27-alpine

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
