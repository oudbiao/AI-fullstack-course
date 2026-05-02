FROM node:20-alpine AS builder

WORKDIR /app

# 增加 Node.js 内存限制
ENV NODE_OPTIONS="--max-old-space-size=2048"

# 先安装依赖，避免每次只改文档或图片都重新安装 node_modules
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

# 再复制项目文件，最大化 Docker 构建缓存命中
COPY docusaurus.config.js ./
COPY sidebars.js ./
COPY scripts ./scripts
COPY docs ./docs
COPY i18n ./i18n
COPY src ./src
COPY static ./static

# 构建应用
RUN npm run build

FROM nginx:1.27-alpine

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
