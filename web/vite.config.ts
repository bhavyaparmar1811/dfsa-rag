/// <reference types="vitest/config" />
import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// The browser calls /api/*; the dev server forwards it to FastAPI and adds the
// X-API-Key header itself, so the key never reaches the JS bundle. API_KEY comes
// from web/.env.local if set, else from the project's .env (the one FastAPI reads).
export default defineConfig(({ mode }) => {
  const root = loadEnv(mode, path.resolve(__dirname, ".."), "");
  const local = loadEnv(mode, __dirname, "");
  const apiKey = local.API_KEY || root.API_KEY || "";
  const target = local.API_URL || "http://127.0.0.1:8000";
  const headers: Record<string, string> = apiKey ? { "X-API-Key": apiKey } : {};

  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target,
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/api/, ""),
          headers,
        },
      },
    },
    test: {
      environment: "jsdom",
      include: ["src/**/*.test.{ts,tsx}"],
    },
  };
});
