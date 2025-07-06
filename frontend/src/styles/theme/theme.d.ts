import type { TimelineConnectorClassKey, TimelineConnectorProps } from "@mui/lab/TimelineConnector";
import type { ComponentsOverrides, ComponentsProps, ComponentsVariants } from "@mui/material/styles";

declare module "@mui/material/styles" {
	interface Color {
		50: string;
		100: string;
		200: string;
		300: string;
		400: string;
		500: string;
		600: string;
		700: string;
		800: string;
		900: string;
		950: string;
	}

	type PartialColor = Partial<Color>;

	interface PaletteColor {
		activated: string;
		hovered: string;
		selected: string;
	}

	interface SimplePaletteColorOptions {
		activated?: string;
		hovered?: string;
		selected?: string;
	}

	interface Palette {
		neutral: PartialColor;
		shadow: string;
		Backdrop: { bg: string };
		OutlinedInput: { border: string };
	}

	interface PaletteOptions {
		neutral?: PartialColor;
		shadow?: string;
		Backdrop?: { bg?: string };
		OutlinedInput?: { border?: string };
	}

	interface TypeBackground {
		level1: string;
		level2: string;
		level3: string;
	}

	interface Components<Theme = unknown> {
		MuiTimelineConnector: {
			defaultProps?: ComponentsProps["MuiTimelineConnector"];
			styleOverrides?: ComponentsOverrides<Theme>["MuiTimelineConnector"];
			variants?: ComponentsVariants<Theme>["MuiTimelineConnector"];
		};
	}

	interface ComponentsPropsList {
		MuiTimelineConnector: TimelineConnectorProps;
	}

	interface ComponentNameToClassKey {
		MuiTimelineConnector: TimelineConnectorClassKey;
	}
}

declare module "@mui/material/Chip" {
	interface ChipPropsVariantOverrides {
		soft: true;
	}

	interface ChipClasses {
		soft: string;
		softPrimary: string;
		softSecondary: string;
		softSuccess: string;
		softInfo: string;
		softWarning: string;
		softError: string;
	}
}
