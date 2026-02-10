# Nginx 代理配置指南

## 域名配置

三个域名**都直接指向同一服务**（不互相跳转）：

| 域名 | 用途 |
|------|------|
| `learning.airoads.org` | 直接代理到课程站点 |
| `www.airoads.org` | 直接代理到课程站点 |
| `airoads.org` | 直接代理到课程站点 |

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
# 三个域名均应直接返回 200（同一站点）
curl -I https://learning.airoads.org
curl -I https://www.airoads.org
curl -I https://airoads.org
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

### 问题：访问时提示「证书不可信」/ Certificate not trusted

**1. 使用 Cloudflare 时**

- 访客必须通过域名访问：**https://learning.airoads.org**，不要用服务器 IP 或「直接解析到源站」的域名访问。
- 在 Cloudflare 控制台：**SSL/TLS → 概述** 选择 **「完全」** 或 **「完全(严格)」**，不要用「灵活」。
- DNS 里该域名的代理状态应为 **已代理（橙色云朵）**，这样流量才走 Cloudflare，浏览器拿到的是 Cloudflare 的可信证书。若解析到源站 IP，浏览器会看到源站证书（如 Origin Certificate），会报不可信。

**2. 未使用 Cloudflare 时**

- 源站必须使用**公信 CA 签发的证书**（浏览器才会信任）。推荐用 **Let's Encrypt**（免费）：
  ```bash
  # 在服务器上安装 certbot，示例（Ubuntu）
  sudo apt install certbot
  sudo certbot certonly --standalone -d learning.airoads.org
  # 证书一般在 /etc/letsencrypt/live/learning.airoads.org/
  ```
- 在 `airoads.conf` 里把 `ssl_certificate` / `ssl_certificate_key` 指向上述路径，重载 Nginx。

**3. 其他检查**

- 确保证书未过期，且覆盖当前访问的域名（如 `*.airoads.org`、`airoads.org`）。
- 若使用 Cloudflare Origin Certificate，仅用于「Cloudflare → 源站」这一段，访客应始终通过 Cloudflare 访问，才会看到可信证书。

### 问题：SSL 证书错误（域名/过期）

确保证书覆盖所有需要的域名（`*.airoads.org` + `airoads.org`），且未过期。

### 问题：重定向循环

如果 Cloudflare 的 SSL 模式设为 "Flexible"，会导致循环。请设置为 **"Full"** 或 **"Full (Strict)"**。

## 相关文件

- Nginx 配置：[nginx/airoads.conf](nginx/airoads.conf)
- Docusaurus 配置：[docusaurus.config.js](docusaurus.config.js)
- Docker Compose：[docker-compose.yml](docker-compose.yml)
