// import { defineConfig } from "vite";
// import react from "@vitejs/plugin-react";

// export default defineConfig({
//   base: "/",
//   plugins: [react()],
//   preview: {
//     port: 5173,
//   },
//   server: {
//     proxy: {
//       "/api": {
//         target: "http://10.245.240.33:9100",
//         changeOrigin: true,
//         secure: false,
//       },
//       "/ai-api": {
//         target: "http://10.245.240.33:9101",
//         changeOrigin: true,
//         secure: false,
//       },
//       "/v1": {
//         target: "http://10.245.240.33:8002",
//         changeOrigin: true,
//         secure: false,
//       },
//             "/v2": {
//         target: "http://10.245.240.33:8003",
//         changeOrigin: true,
//         secure: false,
//       },
//     },
//   },
// });
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  const d365Target = env.VITE_D365_PROXY_TARGET || "http://localhost:9100";
  const aiTarget = env.VITE_AI_API_PROXY_TARGET || "http://localhost:9101";
  const aiBot = env.VITE_AI_API_BOT_PROXY_TARGET || "http://localhost:8002";
  const aiEmail = env.VITE_AI_API_EMAIL_PROXY_TARGET || "http://localhost:8003";
  const enableHttps = env.ENABLE_HTTPS === "true";

  return {
    plugins: [react()],
    server: {
      https: enableHttps,
      proxy: {
        // D365 Sales
        "/api": {
          target: d365Target,
          changeOrigin: true,
        },
        // Lenovo AIBackend (after /ai-api deploy)
        "/ai-api": {
          target: aiTarget,
          changeOrigin: true,
          secure: false,
        },
        "/v1": {
          target: aiBot,
          changeOrigin: true,
          secure: false,
        },
        "/v2": {
          target: aiEmail,
          changeOrigin: true,
          secure: false,
        },
      },
    },
  };
});
