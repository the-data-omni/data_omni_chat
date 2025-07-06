import { appConfig } from "@/config/app";
import { createLogger } from "@/lib/logger";

export const logger = createLogger({ level: appConfig.logLevel });
