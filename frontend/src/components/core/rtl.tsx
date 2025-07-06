"use client";

import * as React from "react";
import createCache from "@emotion/cache";
import type { EmotionCache } from "@emotion/cache";
import { CacheProvider } from "@emotion/react";
import stylisRTLPlugin from "stylis-plugin-rtl";

import { appConfig } from "@/config/app";
import { useSettings } from "@/components/core/settings/settings-context";

export type Direction = "ltr" | "rtl";

function styleCache(): EmotionCache {
	return createCache({ key: "mui-rtl", prepend: true, stylisPlugins: [stylisRTLPlugin] });
}

export interface RTLProps {
	children: React.ReactNode;
}

export function Rtl({ children }: RTLProps): React.JSX.Element {
	const { settings } = useSettings();
	const direction = settings.direction ?? appConfig.direction;

	React.useEffect(() => {
		document.dir = direction;
	}, [direction]);

	if (direction === "rtl") {
		return <CacheProvider value={styleCache()}>{children}</CacheProvider>;
	}

	return <React.Fragment>{children}</React.Fragment>;
}
