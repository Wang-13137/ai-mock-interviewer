# syntax=docker/dockerfile:1
# AI 模拟面试官 - 单容器版
#
# 刻意的取舍：基础镜像不装 torch / sentence-transformers（约 2GB），
# 容器内检索走 BM25（零依赖默认策略），需要语义检索时把 models/ 挂进来并
# 换用 -slim 之外的自定义镜像。小镜像（<300MB）比「全都有」更重要。

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config

# 非 root 运行
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# 密钥一律走环境变量注入：docker run -e ZHIPU_API_KEY=... -e AIMI_LLM__PROVIDER=zhipu
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["python", "-m", "uvicorn", "src.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
