function withProtocol(url: string): string {
	return url.startsWith("http") ? url : `https://${url}`;
}

export function getAppUrl(): URL {
	// In production you should set this to your app URL.
	if (import.meta.env.VITE_APP_URL) {
		return new URL(withProtocol(import.meta.env.VITE_APP_URL));
	}

	// Vercel does not automatically set VITE_VERCEL_URL.
	// You can set it manually in your Vercel project settings as:
	// VITE_VERCEL_URL = ${VERCEL_URL}
	if (import.meta.env.VITE_VERCEL_URL) {
		return new URL(withProtocol(import.meta.env.VITE_VERCEL_URL));
	}

	// Since this runs entirely in the browser, we use `location` to get the current URL.
	// eslint-disable-next-line unicorn/prefer-global-this
	return new URL(window.location.origin);
}
