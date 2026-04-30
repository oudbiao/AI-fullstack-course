FROM node:20-alpine

WORKDIR /app

# 增加 Node.js 内存限制
ENV NODE_OPTIONS="--max-old-space-size=2048"

# 先安装依赖，避免每次只改文档或图片都重新安装 node_modules
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

# 再复制项目文件，最大化 Docker 构建缓存命中
COPY docusaurus.config.js ./
COPY sidebars.js ./
COPY docs ./docs
COPY src ./src
COPY static ./static

# 构建应用
RUN npm run build

EXPOSE 3000

# 启动应用
CMD ["npm", "run", "serve", "--", "--host", "0.0.0.0", "--port", "3000"]
