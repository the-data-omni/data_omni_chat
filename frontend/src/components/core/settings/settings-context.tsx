"use client";

import * as React from "react";

import type { Settings } from "@/types/settings";

export interface SettingsContextValue {
	settings: Settings;
	setSettings: (settings: Settings) => void;
}

export const SettingsContext = React.createContext<SettingsContextValue>({
	settings: {},
	setSettings: () => {
		// noop
	},
});

export interface SettingsProviderProps {
	children: React.ReactNode;
	settings: Settings;
}

export function SettingsProvider({ children, settings: initialSettings }: SettingsProviderProps): React.JSX.Element {
	const [settings, setSettings] = React.useState<Settings>(initialSettings);

	React.useEffect(() => {
		setSettings(initialSettings);
	}, [initialSettings]);

	return <SettingsContext.Provider value={{ settings, setSettings }}>{children}</SettingsContext.Provider>;
}

export const SettingsConsumer = SettingsContext.Consumer;

export function useSettings(): SettingsContextValue {
	const context = React.useContext(SettingsContext);

	if (!context) {
		throw new Error("useSettings must be used within a SettingsProvider");
	}

	return context;
}
