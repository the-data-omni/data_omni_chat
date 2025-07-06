import * as React from "react";
import { Navigate, type RouteObject } from "react-router-dom";

import { route as dashboardRoute } from "./dashboard";

export const routes: RouteObject[] = [
	{ index: true, element: <Navigate to="/chat" replace /> },
	{ path: "dashboard", element: <Navigate to="/chat" replace /> },
	{
		path: "errors",
		children: [
			{
				path: "internal-server-error",
				lazy: async () => {
					const { Page } = await import("@/pages/errors/internal-server-error");
					return { Component: Page };
				},
			},
			{
				path: "not-authorized",
				lazy: async () => {
					const { Page } = await import("@/pages/errors/not-authorized");
					return { Component: Page };
				},
			},
			{
				path: "not-found",
				lazy: async () => {
					const { Page } = await import("@/pages/errors/not-found");
					return { Component: Page };
				},
			},
		],
	},
	dashboardRoute,
	{ path: "*", element: <Navigate to="/chat/direct/TRD-LLM" replace /> },
];
