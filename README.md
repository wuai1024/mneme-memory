# Mneme Memory Service

[![Build & Release](https://github.com/wuai1024/mneme-memory/actions/workflows/release.yml/badge.svg)](https://github.com/wuai1024/mneme-memory/actions/workflows/release.yml)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/wuai1024/mneme-memory?sort=semver)](https://github.com/wuai1024/mneme-memory/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**语义持久化记忆服务** — 为 AI Agent 构建长期记忆的轻量解决方案。

支持以「主体 → 谓词 → 对象」三分量存储事实，结合 sentence-transformers 向量嵌入和 SQLite，实现语义搜索与关键词检索双轨并行。

---

## 特性

- **语义搜索** — 基于 `all-MiniLM-L6-v2` 向量模型，余弦相似度召回
- **关键词搜索** — 基于 SQLite FTS5 全文检索
- **混合搜索** — 语义 + 关键词双轨并行，结果去重合并
- **三分量事实存储** — `Subject / Predicate / Object` 结构化记忆，易于推理
- **会话摘要** — 对话级别摘要存储，支持跨会话上下文检索
- **批量操作** — 支持批量创建/删除事实和摘要
- **数据导出/导入** — 支持全量数据备份和恢复
- **监控指标** — Prometheus 格式指标端点
- **API Key 认证** — 全 API 无状态认证，适合内网部署
- **单文件 SQLite** — 无外部数据库依赖，备份即备份一个 `.db` 文件
- **轻量部署** — CPU-only，支持 x86_64 / arm64 双架构，镜像体积小（< 300MB），任何机器直接运行

---

## 快速开始

### 1. 启动服务

```bash
# 拉取并运行
docker run -d \
  --name mneme-memory \
  -p 33333:33333 \
  -v $(pwd)/data:/data \
  -e MEMORY_API_KEY="your-secret-key-here" \
  ghcr.io/wuai1024/mneme-memory:latest
```

> `MEMORY_API_KEY` 为必填项，所有 API 请求均需携带 `X-API-Key` Header。

### 2. 验证服务

```bash
curl http://localhost:33333/health
# {"status":"ok","embedding_model":"all-MiniLM-L6-v2","embedding_dimension":384,"version":"1.1.0"}
```

### 3. 写入一条记忆

```bash
curl -X POST http://localhost:33333/facts \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"subject":"张三","predicate":"住在","object":"上海"}'
```

### 4. 语义搜索

```bash
curl -X POST http://localhost:33333/search \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query":"张三住在哪里？","top_k":3}'
```

---

## API 文档

### 核心接口

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| `GET` | `/health` | 健康检查 | ❌ |
| `POST` | `/facts` | 新增事实 | ✅ |
| `POST` | `/facts/batch` | 批量新增事实 | ✅ |
| `GET` | `/facts` | 列举事实 | ✅ |
| `GET` | `/facts/{id}` | 获取单条 | ✅ |
| `PUT` | `/facts/{id}` | 更新事实 | ✅ |
| `DELETE` | `/facts/{id}` | 删除事实 | ✅ |
| `DELETE` | `/facts/batch` | 批量删除事实 | ✅ |
| `POST` | `/summaries` | 新增摘要 | ✅ |
| `POST` | `/summaries/batch` | 批量新增摘要 | ✅ |
| `GET` | `/summaries` | 列举摘要 | ✅ |
| `DELETE` | `/summaries/{id}` | 删除摘要 | ✅ |
| `POST` | `/search` | 搜索 | ✅ |

### 数据管理接口

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| `GET` | `/export` | 导出所有数据 | ✅ |
| `POST` | `/import` | 导入数据 | ✅ |
| `GET` | `/metrics` | 获取监控指标 | ❌ |
| `POST` | `/maintenance/vacuum` | 数据库整理 | ✅ |

### 认证方式

所有需要认证的请求，必须携带 Header：

```
X-API-Key: your-secret-key-here
```

无 key 或 key 错误返回 `401 Unauthorized`。

### 搜索模式

`/search` 接口支持三种搜索模式：

```json
{
  "query": "张三住在哪里？",
  "top_k": 3,
  "mode": "semantic"  // semantic | keyword | hybrid
}
```

- **semantic** (默认): 纯语义搜索，基于向量相似度
- **keyword**: 纯关键词搜索，基于 FTS5 全文检索
- **hybrid**: 混合搜索，语义 + 关键词结果去重合并

### 批量操作

批量创建事实：

```bash
curl -X POST http://localhost:33333/facts/batch \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "facts": [
      {"subject":"张三","predicate":"住在","object":"上海"},
      {"subject":"李四","predicate":"住在","object":"北京"}
    ]
  }'
```

批量删除事实：

```bash
curl -X DELETE http://localhost:33333/facts/batch \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"ids":["id1","id2","id3"]}'
```

### 更新事实

```bash
curl -X PUT http://localhost:33333/facts/{id} \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"subject":"张三","predicate":"住在","object":"北京"}'
```

---

## 配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MEMORY_API_KEY` | **必填** | API 认证密钥 |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 向量模型名称 |
| `DATA_DIR` | `/data` | SQLite 数据库目录 |
| `PORT` | `33333` | 服务端口 |
| `SKIP_AUTH` | `false` | 设为 `true` 可跳过认证（仅开发用） |
| `CORS_ORIGINS` | `*` | CORS 允许的源，逗号分隔 |
| `MODEL_CACHE` | `~/.cache/huggingface/` | 模型缓存目录 |

---

## 项目结构

```
.
├── app/
│   ├── main.py          # FastAPI 服务主入口
│   ├── database.py      # SQLite + 向量检索（numpy 优化）
│   ├── embedding.py     # sentence-transformers 封装
│   └── models.py        # Pydantic 数据模型
├── Dockerfile           # CPU-only 轻量镜像（支持 x86_64 / arm64）
├── requirements.txt     # Python 依赖
├── docker-compose.yml   # Docker Compose 编排
├── .env.example         # 环境变量模板
└── README.md
```

---

## 本地开发

```bash
# 安装依赖
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 启动服务（开发模式，热重载）
MEMORY_API_KEY=your-secret-key SKIP_AUTH=true uvicorn app.main:app --reload --port 33333

# 运行测试
python test_api.py
```

> 本地开发时需提前下载 embedding 模型（约 90MB），首次启动会自动缓存到 `~/.cache/huggingface/`。

---

## 性能优化

v1.1.0 版本性能优化：

1. **连接池**: 使用线程本地存储的连接池，避免每次请求创建新连接
2. **numpy 向量化**: 余弦相似度计算使用 numpy，性能提升 5-10x
3. **批量操作**: 支持批量创建/删除，减少网络往返
4. **WAL 模式**: SQLite 使用 WAL 日志模式，提升并发性能
5. **索引优化**: 为常用查询字段添加索引

---

## 安全注意

- **生产环境务必设置强健的 `MEMORY_API_KEY`**（建议 32 字节以上随机字符串）
- `SKIP_AUTH=true` 仅限本地开发，不要在公网或内网生产环境使用
- 数据库文件包含所有记忆内容，注意文件权限控制
- CORS 默认允许 `*`，生产环境请通过 `CORS_ORIGINS` 环境变量限制
- API 密钥比较使用 `hmac.compare_digest`，防止时序攻击

---

## License

MIT
