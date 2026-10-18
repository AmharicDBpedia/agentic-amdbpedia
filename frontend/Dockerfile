# Frontend image: build the SvelteKit app with adapter-node, then run the
# produced Node server. PUBLIC_CROSS_LINGUAL_URL is read at *runtime* via
# $env/dynamic/public, so it is NOT needed at build time -- set it on the
# container instead and no rebuild is required to repoint the API URL.
FROM node:22-slim AS build

WORKDIR /app
RUN corepack enable

# Dependency layer first for caching.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

COPY . .
RUN pnpm run build && pnpm prune --prod

# Runtime image: only the built server + production node_modules.
FROM node:22-slim AS runtime

WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000

COPY --from=build /app/build ./build
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./package.json

EXPOSE 3000

CMD ["node", "build"]
