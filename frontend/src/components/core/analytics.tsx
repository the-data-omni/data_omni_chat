"use client";

import * as React from "react";
import { GTMProvider, useGTMDispatch } from "react-gtm-hook";
import { useSearchParams } from "react-router-dom";

import { usePathname } from "@/hooks/use-pathname";

export interface PageViewTrackerProps {
	children: React.ReactNode;
}

function PageViewTracker({ children }: PageViewTrackerProps): React.JSX.Element {
	const dispatch = useGTMDispatch();
	const pathname = usePathname();
	const [searchParams] = useSearchParams();

	React.useEffect(() => {
		dispatch({ event: "page_view", page: pathname });
	}, [dispatch, pathname, searchParams]);

	return <React.Fragment>{children}</React.Fragment>;
}

export interface AnalyticsProps {
	children: React.ReactNode;
}

/**
 * This loads GTM and tracks the page views.
 *
 * If GTM ID is not configured, this will no track any event.
 */
export function Analytics({ children }: AnalyticsProps): React.JSX.Element {
	if (!import.meta.env.VITE_GOOGLE_TAG_MANAGER_ID) {
		return <React.Fragment>{children}</React.Fragment>;
	}

	return (
		<GTMProvider config={{ id: import.meta.env.VITE_GOOGLE_TAG_MANAGER_ID }}>
			<PageViewTracker>{children}</PageViewTracker>
		</GTMProvider>
	);
}
