# UAII Dashboard — self-host image (API + built web UI)
# Expects a built `uaii` binary mounted or copied at build time.

FROM node:20-bookworm-slim AS webbuild
WORKDIR /app
COPY package.json package-lock.json* ./
COPY web/package.json web/package-lock.json* ./web/
RUN npm install && npm install --prefix web
COPY web ./web
RUN npm run build --prefix web

FROM node:20-bookworm-slim
WORKDIR /app
ENV NODE_ENV=production \
    UAII_DASH_BIND=0.0.0.0 \
    UAII_DASH_PORT=8787 \
    UAII_MODEL_DIR=/data/models \
    UAII_BIN=/opt/uaii/uaii

COPY package.json package-lock.json* ./
RUN npm install --omit=dev
COPY server ./server
COPY --from=webbuild /app/web/dist ./web/dist
COPY uaii-dash.example.json ./uaii-dash.example.json

RUN mkdir -p /data/models
VOLUME ["/data/models"]
EXPOSE 8787

# Mount host-built binary: -v /path/to/uaii:/opt/uaii/uaii:ro
CMD ["node", "server/index.mjs"]
