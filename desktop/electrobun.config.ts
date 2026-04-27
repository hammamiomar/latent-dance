import type { ElectrobunConfig } from "electrobun";

export default {
  app: {
    name: "hambajuba2ba",
    identifier: "com.hambajuba2ba.desktop",
    version: "0.0.1",
  },
  build: {
    bun: {
      entrypoint: "src/bun/index.ts",
    },
    copy: {
      "src/bun/libMacWindowEffects.dylib": "bun/libMacWindowEffects.dylib",
    },
  },
} satisfies ElectrobunConfig;
