import * as React from "react";
import Avatar from "@mui/material/Avatar";
import AvatarGroup from "@mui/material/AvatarGroup";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import {
    GearIcon,
    FilePptIcon,
    PresentationChartIcon,
    CheckCircleIcon, // <-- Icon for Success
    ClockIcon,             // <-- Icon for Verifying
    WarningCircleIcon // <-- Icon for Error/Unknown
} from "@phosphor-icons/react/dist/ssr";
import  Chip  from '@mui/material/Chip'; // <-- Import Chip

// --- Import the new component ---
import { APISettingsModal, APISettings } from '../dashboard/api-settings-modal';

import { usePopover } from "@/hooks/use-popover";

import type { Thread } from "./types";
import dayjs from "dayjs";

const user = {
    id: "USR-000",
    name: "Sofia Rivers",
    avatar: "/assets/avatar.png",
    email: "sofia@thedataomni.io",
} as const;

export interface ThreadToolbarProps {
    thread: Thread;
	onExport: () => void; // --- ⬇️ ADD THIS PROP ---
    onExportToGoogleSlides: () => void;
    onSettingsSave: (settings: APISettings) => void;
    apiSettings: APISettings;
    connectionStatus: 'unknown' | 'verifying' | 'success' | 'error';
    modelName: string;
}

// --- DEFINE COLORS AND MESSAGES FOR THE INDICATOR ---
const statusMap = {
    unknown: { color: 'action.disabled', message: 'API settings not configured' },
    verifying: { color: 'warning.main', message: 'Verifying connection...' },
    success: { color: 'success.main', message: 'Connection successful' },
    error: { color: 'error.main', message: 'Connection failed' },
};

export function ThreadToolbar({ thread, onExport,onExportToGoogleSlides, onSettingsSave, apiSettings, connectionStatus, modelName }: ThreadToolbarProps): React.JSX.Element {
    const popover = usePopover<HTMLButtonElement>();

    const recipients = (thread.participants ?? []).filter((participant) => participant.id !== user.id);
    const [isSettingsOpen, setIsSettingsOpen] = React.useState(false);
    const { color, message } = statusMap[connectionStatus];

        const getStatusIndicator = () => {
        switch (connectionStatus) {
            case 'success': {
                return {
                    icon: <CheckCircleIcon size={18} weight="fill" />,
                    color: 'success' as 'success',
                    label: modelName,
                    tooltip: 'Connection successful'
                };
            }
            case 'verifying': {
                return {
                    icon: <ClockIcon size={18} />,
                    color: 'warning' as 'warning',
                    label: 'Verifying...',
                    tooltip: 'Verifying connection...'
                };
            }
            case 'error': {
                return {
                    icon: <WarningCircleIcon size={18} weight="fill" />,
                    color: 'error' as 'error',
                    label: 'Connection Failed',
                    tooltip: 'Connection failed. Check settings.'
                };
            }
            
            default: {
                return {
                    icon: <WarningCircleIcon size={18} />,
                    color: 'default' as 'default',
                    label: 'API Not Set',
                    tooltip: 'API settings not configured'
                };
            }
        }
    };

    const statusIndicator = getStatusIndicator();

    return (
        <React.Fragment>
            <APISettingsModal
                open={isSettingsOpen}
                onClose={() => setIsSettingsOpen(false)}
                onSave={onSettingsSave}
                currentSettings={apiSettings}
            />
            <Stack
                direction="row"
                spacing={2}
                sx={{
                    alignItems: "center",
                    borderBottom: "1px solid var(--mui-palette-divider)",
                    flex: "0 0 auto",
                    justifyContent: "space-between",
                    minHeight: "64px",
                    px: 2,
                    py: 1,
                }}
            >
                <Stack direction="row" spacing={2} sx={{ alignItems: "center", minWidth: 0 }}>
                    <AvatarGroup
                        max={2}
                        sx={{
                            "& .MuiAvatar-root": {
                                fontSize: "var(--fontSize-xs)",
                                ...(thread.type === "group"
                                    ? { height: "24px", ml: "-16px", width: "24px", "&:nth-of-type(2)": { mt: "12px" } }
                                    : { height: "36px", width: "36px" }),
                            },
                        }}
                    >
                        {recipients.map((recipient) => (
                            <Avatar key={recipient.id} src={recipient.avatar} />
                        ))}
                    </AvatarGroup>
                    <Box sx={{ minWidth: 0 }}>
                        <Typography noWrap variant="subtitle2">
                            {recipients.map((recipient) => recipient.name).join(", ")}
                        </Typography>
                        {thread.type === "direct" ? (
                            <Typography color="text.secondary" variant="caption">
                                Active at {dayjs().format("h:mm A")}
                            </Typography>
                        ) : null}
                    </Box>
                </Stack>
                <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                    <Tooltip title={statusIndicator.tooltip}>
                        <Chip
                            icon={statusIndicator.icon}
                            label={statusIndicator.label}
                            color={statusIndicator.color}
                            size="small"
                            variant="outlined"
                            sx={{
                                // Make the success indicator stand out more
                                ...(connectionStatus === 'success' && {
                                    bgcolor: 'success.lightest',
                                    fontWeight: 600,
                                }),
                            }}
                        />
                    </Tooltip>
                    <Tooltip title="API Settings">
                        <IconButton onClick={() => setIsSettingsOpen(true)}>
                            <GearIcon />
                        </IconButton>
                    </Tooltip>

					{/* --- ⬇️ ADD THIS EXPORT BUTTON --- */}
					<Tooltip title="Export to Google Slides">
                        <IconButton onClick={onExportToGoogleSlides}>
                            <PresentationChartIcon />
                        </IconButton>
                    </Tooltip>
                    <Tooltip title="Export to PowerPoint">
                        <IconButton onClick={onExport}>
                            <FilePptIcon />
                        </IconButton>
                    </Tooltip>
                </Stack>
            </Stack>
        </React.Fragment>
    );
}
