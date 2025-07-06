import type { NavItemConfig } from "@/types/nav";
import { paths } from "@/paths";

// NOTE: We did not use React Components for Icons, because
//  you may want to get the config from the server.

// NOTE: First level of navItem elements are groups.

export interface DashboardConfig {
	// Overriden by Settings Context.
	layout: "horizontal" | "vertical";
	// Overriden by Settings Context.
	navColor: "blend_in" | "discrete" | "evident";
	navItems: NavItemConfig[];
}

export const dashboardConfig = {
	layout: "vertical",
	navColor: "evident",
	navItems: [

				{
			key: "chat",
			title: "Chat",
			items: [{ key: "chat", title: "Chat", href: paths.home, icon: "file-dashed" }],
		},
	],
} satisfies DashboardConfig;
