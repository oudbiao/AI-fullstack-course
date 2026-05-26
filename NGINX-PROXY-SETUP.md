# Nginx 代理配置指南

## 域名配置

生产环境只保留一个 canonical 域名：`airoads.org`。

| 域名 | 用途 |
|------|------|
| `airoads.org` | 直接代理到课程站点 |
| `learning.airoads.org` | 301 跳转到 `https://airoads.org$request_uri` |
| `www.airoads.org` | 301 跳转到 `https://airoads.org$request_uri` |

不要把 `learning.airoads.org`、`www.airoads.org` 和 `airoads.org` 写在同一个 `server_name` 里一起代理。这样三个域名都会返回 `200`，Google 会看到三份重复页面，Search Console 容易出现“备用网页”“重复网页”“Google 选择的规范网页与用户指定的不同”等提示。

## 配置文件

| 文件 | 说明 |
|------|------|
| `nginx/airoads.conf` | 外层生产 Nginx 代理配置 |
| `docker/nginx.conf` | `ai-course` 应用容器内的静态站点 Nginx 配置 |

当前部署有两层 Nginx：

1. 外层 Nginx 容器：接收公网 HTTPS，负责 canonical 域名跳转和反向代理。
2. `ai-course` 应用容器内的 Nginx：服务 Astro 生成的静态文件，并兜底处理旧路径和 404。

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

### 2. 确认 Docker 网络

外层 Nginx 容器和课程容器必须在同一个 `proxy-net` 网络里：

```bash
docker network inspect proxy-net >/dev/null 2>&1 || docker network create proxy-net
docker network connect proxy-net <nginx-container-name> 2>/dev/null || true
```

`ai-course` 容器会通过项目里的 `docker-compose.yml` 自动加入 `proxy-net`。

### 3. 更新外层 Docker Nginx 配置

如果外层 Nginx 容器把配置目录挂载到了宿主机，直接覆盖挂载文件：

```bash
cp /opt/ai-course/nginx/airoads.conf /path/to/mounted/conf.d/airoads.conf
docker exec <nginx-container-name> nginx -t
docker exec <nginx-container-name> nginx -s reload
```

如果没有挂载配置目录，可以直接复制到容器里：

```bash
docker cp /opt/ai-course/nginx/airoads.conf <nginx-container-name>:/etc/nginx/conf.d/airoads.conf
docker exec <nginx-container-name> nginx -t
docker exec <nginx-container-name> nginx -s reload
```

如果外层 Nginx 也是 Compose 管理，推荐用只读挂载固定配置：

```yaml
services:
  nginx:
    image: nginx:1.27-alpine
    volumes:
      - /opt/ai-course/nginx/airoads.conf:/etc/nginx/conf.d/airoads.conf:ro
      - /etc/nginx/certs:/etc/nginx/certs:ro
    networks:
      - proxy-net

networks:
  proxy-net:
    external: true
```

### 4. 验证

```bash
curl -I https://airoads.org/zh-cn/
# 期望：HTTP/2 200

curl -I https://learning.airoads.org/zh-cn/
# 期望：HTTP/2 301
# Location: https://airoads.org/zh-cn/

curl -I https://www.airoads.org/zh-cn/
# 期望：HTTP/2 301
# Location: https://airoads.org/zh-cn/

curl -I https://airoads.org/zh-Hans/
# 期望：HTTP/2 301
# Location: /zh-cn/

curl -I https://airoads.org/not-real-page-for-status-test
# 期望：HTTP/2 404
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

### 问题：Search Console 报重复网页或备用网页

检查外层 Nginx 是否把三个域名写在同一个 `server_name` 里并直接代理：

```nginx
server_name learning.airoads.org www.airoads.org airoads.org;
```

这种写法不适合当前站点 SEO。应该让 `learning.airoads.org` 和 `www.airoads.org` 单独 `301` 到 `airoads.org`，只让 `airoads.org` 代理到课程服务。

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
- Astro Starlight 配置：[astro.config.mjs](astro.config.mjs)
- Docker Compose：[docker-compose.yml](docker-compose.yml)
