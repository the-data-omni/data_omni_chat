import * as React from "react";
import { Outlet, Navigate } from "react-router-dom";
import type { RouteObject } from "react-router-dom";
import { Layout as ChatLayout } from "@/components/chat/layout";

export const route: RouteObject = {
  path: "chat",
  element: (
    <ChatLayout>
      <Outlet />
    </ChatLayout>
  ),
  children: [
    // Redirect the index ("/chat") to "/chat/direct/TRD-LLM"
    {
      index: true,
      element: <Navigate to="direct/TRD-LLM" replace />,
    },
    {
      path: "compose",
      lazy: async () => {
        const { Page } = await import("@/pages/chat/compose");
        return { Component: Page };
      },
    },
    {
      path: ":threadType/:threadId",
      lazy: async () => {
        const { Page } = await import("@/pages/chat/thread");
        return { Component: Page };
      },
    },
  ],
};
