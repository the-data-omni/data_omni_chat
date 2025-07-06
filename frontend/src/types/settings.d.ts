import type { PrimaryColor } from "@/styles/theme/types";

export type Direction = "ltr" | "rtl";

export type DashboardLayout = "horizontal" | "vertical";

export type DashboardNavColor = "blend_in" | "discrete" | "evident";

export interface Settings {
	direction?: Direction;
	language?: string;
	primaryColor?: PrimaryColor;
	dashboardLayout?: DashboardLayout;
	dashboardNavColor?: DashboardNavColor;
}
