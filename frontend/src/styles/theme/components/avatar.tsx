import * as React from "react";
import type { Components } from "@mui/material/styles";
import { UserIcon } from "@phosphor-icons/react/dist/ssr/User";

import type { Theme } from "../types";

export const MuiAvatar = {
	defaultProps: { children: <UserIcon fontSize="var(--Icon-fontSize)" /> },
	styleOverrides: {
		root: {
			"--Icon-fontSize": "var(--icon-fontSize-md)",
			fontSize: "0.875rem",
			fontWeight: 500,
			height: "var(--Avatar-size, 40px)",
			letterSpacing: 0,
			width: "var(--Avatar-size, 40px)",
		},
	},
} satisfies Components<Theme>["MuiAvatar"];
