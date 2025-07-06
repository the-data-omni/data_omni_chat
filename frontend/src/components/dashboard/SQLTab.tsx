
import React from "react";
import { Box, Typography } from "@mui/material";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

interface SQLTabProps {
  sql: string; // The prop name remains "sql" as it's used in the parent component
}

export function SQLTab({ sql }: SQLTabProps) {
  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 1 }}>
        CODE
      </Typography>

      <SyntaxHighlighter
        // --- CHANGE: Set the language to "python" for correct highlighting ---
        language="python"
        style={vscDarkPlus}
        customStyle={{
          borderRadius: 0,
          margin: 0,
        }}
        wrapLines={true}
        showLineNumbers
      >
        {sql}
      </SyntaxHighlighter>
    </Box>
  );
}