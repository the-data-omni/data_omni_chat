// "use client";

// import * as React from "react";
// import Box from "@mui/material/Box";
// import IconButton from "@mui/material/IconButton";
// import { List as ListIcon } from "@phosphor-icons/react/dist/ssr/List";
// import { useNavigate } from "react-router-dom";

// import { paths } from "@/paths";
// import { useMediaQuery } from "@/hooks/use-media-query";
// import { usePathname } from "@/hooks/use-pathname";

// import { ChatContext } from "./chat-context";
// import { Sidebar } from "./sidebar";

// export interface ChatViewProps {
// 	children: React.ReactNode;
// }

// export function ChatView({ children }: ChatViewProps): React.JSX.Element {
// 	const {
// 		contacts,
// 		threads,
// 		messages,
// 		createThread,
// 		openDesktopSidebar,
// 		setOpenDesktopSidebar,
// 		openMobileSidebar,
// 		setOpenMobileSidebar,
// 	} = React.useContext(ChatContext);

// 	const navigate = useNavigate();

// 	const pathname = usePathname();

// 	// The layout does not have a direct access to the current thread id param, we need to extract it from the pathname.
// 	const segments = pathname.split("/").filter(Boolean);
// 	const currentThreadId = segments.length === 4 ? segments.at(-1) : undefined;

// 	const mdDown = useMediaQuery("down", "md");

// 	const handleContactSelect = React.useCallback(
// 		(contactId: string) => {
// 			const threadId = createThread({ type: "direct", recipientId: contactId });

// 			navigate(paths.dashboard.chat.thread("direct", threadId));
// 		},
// 		[navigate, createThread]
// 	);

// 	const handleThreadSelect = React.useCallback(
// 		(threadType: string, threadId: string) => {
// 			navigate(paths.dashboard.chat.thread(threadType, threadId));
// 		},
// 		[navigate]
// 	);

// 	return (
// 		<Box sx={{ display: "flex", flex: "1 1 0", minHeight: 0 }}>
// 			<Sidebar
// 				contacts={contacts}
// 				currentThreadId={currentThreadId}
// 				messages={messages}
// 				onCloseMobile={() => {
// 					setOpenMobileSidebar(false);
// 				}}
// 				onSelectContact={handleContactSelect}
// 				onSelectThread={handleThreadSelect}
// 				openDesktop={openDesktopSidebar}
// 				openMobile={openMobileSidebar}
// 				threads={threads}
// 			/>
// 			<Box sx={{ display: "flex", flex: "1 1 auto", flexDirection: "column", overflow: "hidden" }}>
// 				<Box sx={{ borderBottom: "1px solid var(--mui-palette-divider)", display: "flex", flex: "0 0 auto", p: 2 }}>
// 					<IconButton
// 						onClick={() => {
// 							if (mdDown) {
// 								setOpenMobileSidebar((prev) => !prev);
// 							} else {
// 								setOpenDesktopSidebar((prev) => !prev);
// 							}
// 						}}
// 					>
// 						<ListIcon />
// 					</IconButton>
// 				</Box>
// 				{children}
// 			</Box>
// 		</Box>
// 	);
// }
"use client";

import * as React from "react";
import Box from "@mui/material/Box";
// The IconButton and ListIcon are no longer needed.
// import IconButton from "@mui/material/IconButton";
// import { List as ListIcon } from "@phosphor-icons/react/dist/ssr/List";

// The navigate and pathname hooks might not be needed here if their only purpose was for the sidebar.
// import { useNavigate } from "react-router-dom";
// import { paths } from "@/paths";
// import { usePathname } from "@/hooks/use-pathname";

// useMediaQuery is no longer needed.
// import { useMediaQuery } from "@/hooks/use-media-query";

// ChatContext might still be needed by child components, but the sidebar-specific state is removed.
import { ChatContext } from "./chat-context";

// The Sidebar component import is removed.
// import { Sidebar } from "./sidebar";

export interface ChatViewProps {
    children: React.ReactNode;
}

export function ChatView({ children }: ChatViewProps): React.JSX.Element {
    // All state and context values related to the sidebar have been removed.
    // const {
    //     contacts,
    //     threads,
    //     messages,
    //     createThread,
    //     openDesktopSidebar,
    //     setOpenDesktopSidebar,
    //     openMobileSidebar,
    //     setOpenMobileSidebar,
    // } = React.useContext(ChatContext);

    // This logic was for selecting items in the sidebar, so it's no longer needed here.
    // const navigate = useNavigate();
    // const pathname = usePathname();
    // const segments = pathname.split("/").filter(Boolean);
    // const currentThreadId = segments.length === 4 ? segments.at(-1) : undefined;
    // const mdDown = useMediaQuery("down", "md");

    // const handleContactSelect = React.useCallback(
    //     (contactId: string) => {
    //         const threadId = createThread({ type: "direct", recipientId: contactId });
    //         navigate(paths.dashboard.chat.thread("direct", threadId));
    //     },
    //     [navigate, createThread]
    // );

    // const handleThreadSelect = React.useCallback(
    //     (threadType: string, threadId: string) => {
    //         navigate(paths.dashboard.chat.thread(threadType, threadId));
    //     },
    //     [navigate]
    // );

    return (
        // The outer Box that created the flex layout for the sidebar and content is removed.
        // The main content Box now becomes the root element.
        <Box
            sx={{
                display: "flex",
                flex: "1 1 auto",
                flexDirection: "column",
                overflow: "hidden",
            }}
        >
            {/* The header Box with the toggle button is removed. */}
            {/* <Box sx={{ borderBottom: "1px solid var(--mui-palette-divider)", display: "flex", flex: "0 0 auto", p: 2 }}>
                <IconButton
                    onClick={() => {
                        if (mdDown) {
                            setOpenMobileSidebar((prev) => !prev);
                        } else {
                            setOpenDesktopSidebar((prev) => !prev);
                        }
                    }}
                >
                    <ListIcon />
                </IconButton>
            </Box> */}
            {children}
        </Box>
    );
}