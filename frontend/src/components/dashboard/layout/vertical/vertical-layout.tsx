"use client";

import * as React from "react";
import Box from "@mui/material/Box";
import GlobalStyles from "@mui/material/GlobalStyles";

export interface VerticalLayoutProps {
  children?: React.ReactNode;
}

export function VerticalLayout({ children }: VerticalLayoutProps): React.JSX.Element {
  return (
    <React.Fragment>
      <GlobalStyles
        styles={{
          // reset any default body margins so we truly span edge-to-edge
          body: {
            margin: 0,
            padding: 0,
          },
        }}
      />
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          width: "100vw",
          height: "100vh",       // fill the viewport
          bgcolor: "background.default",
        }}
      >
        <Box
          component="main"
          sx={{
            flex: 1,             // grow to fill parent
            width: "100%",       // full width
            height: "100%",      // full height
            p: 0,                // no padding
            m: 0,                // no margin
            display: "flex",
            flexDirection: "column",
          }}
        >
          {children}
        </Box>
      </Box>
    </React.Fragment>
  );
}
