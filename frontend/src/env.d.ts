/// <reference types="vite/client" />

interface ImportMetaEnv {
	// App
	VITE_APP_URL?: string;
	VITE_VERCEL_URL?: string;
	VITE_LOG_LEVEL?: string;
	VITE_AUTH_STRATEGY?: string;

	// Auth0
	VITE_AUTH0_DOMAIN?: string;
	VITE_AUTH0_CLIENT_ID?: string;

	// Clerk
	VITE_CLERK_PUBLISHABLE_KEY?: string;

	// Cognito
	VITE_COGNITO_AUTHORITY?: string;
	VITE_COGNITO_DOMAIN?: string;
	VITE_COGNITO_CLIENT_ID?: string;

	// Supabase
	VITE_SUPABASE_URL?: string;
	VITE_SUPABASE_PUBLIC_KEY?: string;

	// Mapbox
	VITE_MAPBOX_API_KEY?: string;

	// Google Tag Manager
	VITE_GOOGLE_TAG_MANAGER_ID?: string;
}

interface ImportMeta {
	readonly env: ImportMetaEnv;
}
