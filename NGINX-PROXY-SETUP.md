# Nginx 代理配置指南

## 域名配置

| 域名 | 用途 |
|------|------|
| `learning.airoads.org` | **主站**（Docusaurus 课程站点） |
| `www.airoads.org` | 301 跳转到 `learning.airoads.org` |
| `airoads.org` | 301 跳转到 `learning.airoads.org` |

## 配置文件

| 文件 | 说明 |
|------|------|
| `nginx/airoads.conf` | airoads.org 域名及证书配置 |

## 生产服务器部署步骤

### 1. 上传 SSL 证书

为 `airoads.org` 申请证书（建议 Cloudflare Origin Certificate，覆盖 `*.airoads.org` 和 `airoads.org`），放到服务器：

```bash
# 在服务器上
sudo mkdir -p /etc/nginx/certs
# 上传证书文件
# /etc/nginx/certs/airoads-origin.pem
# /etc/nginx/certs/airoads-origin.key
```

### 2. 复制 Nginx 配置到服务器

```bash
sudo cp /opt/ai-course/nginx/airoads.conf /etc/nginx/conf.d/airoads.conf
```

### 3. 验证并重载 Nginx

```bash
# 检查配置语法
sudo nginx -t

# 重新加载 Nginx
sudo systemctl reload nginx
```

### 4. 验证

```bash
# airoads.org 主站
curl -I https://learning.airoads.org
# 应返回 200

# airoads.org 跳转
curl -I https://www.airoads.org   # 301 → learning.airoads.org
curl -I https://airoads.org      # 301 → learning.airoads.org
```

## Docker 容器网络

容器通过 `proxy-net` 外部网络与 Nginx 通信：

```yaml
# docker-compose.yml
networks:
  proxy-net:
    external: true
```

确保 Nginx 容器也在同一网络中，这样 `proxy_pass http://ai-fullstack-course:3000/` 才能工作。

## 常见问题

### 问题：502 Bad Gateway

1. 确保容器正在运行：`docker-compose ps`
2. 确保容器在 `proxy-net` 网络中
3. 检查防火墙规则

### 问题：SSL 证书错误

确保证书覆盖了所有需要的域名（`*.airoads.org` + `airoads.org`）。

### 问题：重定向循环

如果 Cloudflare 的 SSL 模式设为 "Flexible"，会导致循环。请设置为 **"Full"** 或 **"Full (Strict)"**。

## 相关文件

- Nginx 配置：[nginx/airoads.conf](nginx/airoads.conf)
- Docusaurus 配置：[docusaurus.config.js](docusaurus.config.js)
- Docker Compose：[docker-compose.yml](docker-compose.yml)
