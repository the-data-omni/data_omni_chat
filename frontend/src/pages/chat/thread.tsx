
import * as React from "react";
import { Helmet } from "react-helmet-async";
import { useParams } from "react-router-dom";

import type { Metadata } from "@/types/metadata";
import { appConfig } from "@/config/app";
import { AIThreadView } from "@/components/chat/AIThreadView"; // <--- NEW
import type { ThreadType } from "@/components/chat/types";

const metadata = { title: `Thread | Chat | Dashboard | ${appConfig.name}` } satisfies Metadata;

export function Page(): React.JSX.Element {
	console.log("Rendering <Page>...");
  const { threadId, threadType } = useParams() as { threadId: string; threadType: ThreadType };


  return (
    <React.Fragment>
      <Helmet>
        <title>{metadata.title}</title>
      </Helmet>
        <AIThreadView threadId={threadId} threadType={threadType} />
    </React.Fragment>
  );
}
