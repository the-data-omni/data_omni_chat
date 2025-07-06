import type { Settings } from "@/types/settings";
import { logger } from "@/lib/default-logger";

/**
 * Store settings in client's localStorage.
 * This should be used in Client Components.
 *
 * To remove a specific key, set its value to `null`.
 */
export function setSettings(settings: Partial<Settings>): void {
	globalThis.localStorage.setItem("settings", JSON.stringify(settings));
}

/*
 * Retrieve the settings from client's localStorage.
 * This should be used in Client Components.
 */
export function getSettings(): Settings {
	const settingsStr: string | null = globalThis.localStorage.getItem("settings");
	let settings: Settings;

	if (settingsStr) {
		try {
			settings = JSON.parse(settingsStr) as Settings;
		} catch {
			logger.error("Unable to parse the settings");
		}
	}

	settings ||= {};

	return settings;
}
