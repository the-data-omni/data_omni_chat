"use client";

import * as React from "react";
import { use } from "i18next";
import { initReactI18next, useTranslation } from "react-i18next";

import { appConfig } from "@/config/app";
import { logger } from "@/lib/default-logger";
import { de } from "@/locales/de";
import { en } from "@/locales/en";
import { es } from "@/locales/es";

import { useSettings } from "./settings/settings-context";

// eslint-disable-next-line react-hooks/rules-of-hooks
use(initReactI18next)
	.init({
		debug: false,
		ns: Object.keys(en),
		defaultNS: "common",
		fallbackNS: "common",
		fallbackLng: "en",
		resources: {
			de,
			en,
			es,
		},
	})
	// eslint-disable-next-line unicorn/prefer-top-level-await
	.catch((error: unknown) => {
		logger.error("[I18nProvider] Failed to initialize i18n", error);
	});

export interface I18nProviderProps {
	children: React.ReactNode;
}

export function I18nProvider({ children }: I18nProviderProps): React.JSX.Element {
	const { i18n } = useTranslation();
	const { settings } = useSettings();

	const language = settings.language ?? appConfig.language;

	React.useEffect(() => {
		i18n.changeLanguage(language).catch(() => {
			logger.error(`[I18nProvider] Failed to change language to ${language}`);
		});
	}, [i18n, language]);

	return <React.Fragment>{children}</React.Fragment>;
}
