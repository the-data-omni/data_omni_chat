"use client";

import * as React from "react";
import { Helmet, HelmetProvider } from "react-helmet-async";

import "@/styles/global.css";

import type { Metadata } from "@/types/metadata";
import { appConfig } from "@/config/app";
import { getSettings as getPersistedSettings } from "@/lib/settings";
import { ThemeProvider } from "@/components/core//theme-provider";
import { Analytics } from "@/components/core/analytics";
import { I18nProvider } from "@/components/core/i18n-provider";
import { LocalizationProvider } from "@/components/core/localization-provider";
import { Rtl } from "@/components/core/rtl";
import { SettingsButton } from "@/components/core/settings/settings-button";
import { SettingsProvider } from "@/components/core/settings/settings-context";
import { Toaster } from "@/components/core/toaster";

const metadata = { title: appConfig.name } satisfies Metadata;

const AuthProvider: React.FC<React.PropsWithChildren> = React.Fragment;

export interface RootProps {
	children: React.ReactNode;
}

export function Root({ children }: RootProps): React.JSX.Element {
	const settings = getPersistedSettings();

	return (
		<HelmetProvider>
			<Helmet>
				<title>{metadata.title}</title>
				<meta content={appConfig.themeColor} name="theme-color" />
			</Helmet>
			<AuthProvider>
				{/* <Analytics> */}
					<LocalizationProvider>
						<SettingsProvider settings={settings}>
							<I18nProvider>
								<Rtl>
									<ThemeProvider>
										{children}
										{/* <SettingsButton /> */}
										<Toaster position="bottom-right" />
									</ThemeProvider>
								</Rtl>
							</I18nProvider>
						</SettingsProvider>
					</LocalizationProvider>
				{/* </Analytics> */}
			</AuthProvider>
		</HelmetProvider>
	);
}
