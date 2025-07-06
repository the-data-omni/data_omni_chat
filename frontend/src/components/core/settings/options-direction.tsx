"use client";

import type * as React from "react";
import InputLabel from "@mui/material/InputLabel";
import Stack from "@mui/material/Stack";
import { TextAlignLeftIcon } from "@phosphor-icons/react/dist/ssr/TextAlignLeft";
import { TextAlignRightIcon } from "@phosphor-icons/react/dist/ssr/TextAlignRight";

import type { Direction } from "@/types/settings";

import { Option } from "./option";

export interface OptionsDirectionProps {
	onChange?: (value: Direction) => void;
	value?: Direction;
}

export function OptionsDirection({ onChange, value }: OptionsDirectionProps): React.JSX.Element {
	return (
		<Stack spacing={1}>
			<InputLabel>Orientation</InputLabel>
			<Stack direction="row" spacing={2} sx={{ alignItems: "center", flexWrap: "wrap" }}>
				{(
					[
						{ label: "Left-to-right", value: "ltr", icon: <TextAlignLeftIcon /> },
						{ label: "Right-to-left", value: "rtl", icon: <TextAlignRightIcon /> },
					] satisfies { label: string; value: Direction; icon: React.ReactElement }[]
				).map((option) => (
					<Option
						icon={option.icon}
						key={option.label}
						label={option.label}
						onClick={() => {
							onChange?.(option.value);
						}}
						selected={option.value === value}
					/>
				))}
			</Stack>
		</Stack>
	);
}
