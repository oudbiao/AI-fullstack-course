FROM node:20-alpine

WORKDIR /app

# 启用 corepack 来使用 yarn（Node.js 内置）
RUN corepack enable yarn

# 增加 Node.js 内存限制
ENV NODE_OPTIONS="--max-old-space-size=2048"

# 复制项目文件
COPY package*.json ./
COPY docusaurus.config.js ./
COPY sidebars.js ./
COPY docs ./docs
COPY src ./src
COPY static ./static

# 使用 yarn 安装依赖
RUN yarn install --no-optional || npm install

# 构建应用
RUN yarn build || npm run build

EXPOSE 3000

# 启动应用
CMD ["yarn", "serve", "--host", "0.0.0.0", "--port", "3000"]