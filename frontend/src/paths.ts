export const paths = {
	home: "/chat/direct/TRD-LLM",
	dashboard: {
		chat: {
			base: "/dashboard/chat",
			compose: "/dashboard/chat/compose",
			thread: (threadType: string, threadId: string) => `/dashboard/chat/${threadType}/${threadId}`,
		},
	},
	notAuthorized: "/errors/not-authorized",
	notFound: "/errors/not-found",
	internalServerError: "/errors/internal-server-error",
	docs: "https://material-kit-pro-react-docs.thedataomni.io",
	purchase: "https://mui.com/store/items/thedataomni-kit-pro",
} as const;
