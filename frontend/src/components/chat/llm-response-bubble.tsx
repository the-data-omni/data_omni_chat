// "use client";

// import React, { useState, useEffect } from "react";
// import { Card, Stack, Typography, Tabs, Tab, Box } from "@mui/material";
// import type { Message } from "./types";

// // Example tab components that expect "sql" (and build their own data)
// import { ChartTab } from "@/components/dashboard/ChartTab";
// import { DataTab } from "@/components/dashboard/DataTab";
// import { SQLTab } from "@/components/dashboard/SQLTab";
// import { DataFlowTab } from "@/components/dashboard/DataFlowTab";

// interface LLMResponseBubbleProps {
//   message: Message;
//   position: "left" | "right";
// }

// /**
//  * If you want to locally interpret the SQL and build chart data, etc.,
//  * you can do it here with local state. Otherwise, you can just pass the SQL
//  * down to each tab component.
//  */
// function useLocalSqlData(sql?: string) {
//   const [chartData, setChartData] = useState<any[]>([]);
//   const [tableData, setTableData] = useState<any[]>([]);
//   const [dataFlow, setDataFlow] = useState<string>("");

//   useEffect(() => {
//     if (!sql) return;

//     // Example: run query or interpret SQL. We'll just mock some data:
//     setChartData([
//       { date: "2024-11-01", hits: 42 },
//       { date: "2024-11-02", hits: 99 },
//     ]);

//     setTableData([
//       { date: "2024-11-01", visitors: 42 },
//       { date: "2024-11-02", visitors: 99 },
//     ]);

//     setDataFlow(`Local data built from: ${sql}`);
//   }, [sql]);

//   return { chartData, tableData, dataFlow };
// }

// export function LLMResponseBubble({ message, position }: LLMResponseBubbleProps) {
//   // 1. Local tab state so switching tabs doesn't create a new message
//   const [tabValue, setTabValue] = useState(0);

//   // 2. The message has only { content, sql }
//   const { content, sql } = message;

//   // 3. If you want to build chart/table data from sql, do so locally
//   const { chartData, tableData, dataFlow } = useLocalSqlData(sql);

//   return (
//     <Stack
//       direction={position === "right" ? "row-reverse" : "row"}
//       spacing={2}
//       sx={{
//         alignItems: "flex-start",
//         minWidth: "75%", // or whatever size you prefer
//         ml: position === "right" ? "auto" : 0,
//         mr: position === "left" ? "auto" : 0,
//       }}
//     >
//       <Card
//         sx={{
//           px: 2,
//           py: 1,
//           ...(position === "right" && {
//             bgcolor: "var(--mui-palette-primary-main)",
//             color: "var(--mui-palette-primary-contrastText)",
//           }),
//         }}
//       >
//         <Stack spacing={1}>
//           {/* 4. Show LLM text */}
//           <Typography variant="body1">{content}</Typography>

//           {/* 5. Tabs for Chart, Data, SQL, DataFlow */}
//           <Tabs
//             value={tabValue}
//             onChange={(_, newVal) => setTabValue(newVal)}
//             variant="fullWidth"
//             sx={{ borderBottom: "1px solid var(--mui-palette-divider)" }}
//           >
//             <Tab label="Chart" />
//             <Tab label="Data" />
//             <Tab label="SQL" />
//             <Tab label="Data Flow" />
//           </Tabs>

//           <Box sx={{ mt: 1 }}>
//             {tabValue === 0 && <ChartTab data={chartData} />}
//             {tabValue === 1 && <DataTab data={tableData} />}
//             {tabValue === 2 && <SQLTab sql={sql || ""} />}
//             {tabValue === 3 && <DataFlowTab  />}
//           </Box>
//         </Stack>
//       </Card>
//     </Stack>
//   );
// }
"use client";

import React, { useState, useEffect } from "react";
import { Card, Stack, Typography, Tabs, Tab, Box } from "@mui/material";
import type { Message } from "./types";


import { ChartTab } from "@/components/dashboard/ChartTab";
import { SQLTab } from "@/components/dashboard/SQLTab";


interface LLMResponseBubbleProps {
  message: Message; // Now Message includes an optional "data" field with the execution payload data.
  position: "left" | "right";
}

/**
 * Updated hook that accepts the generated SQL/code, the execution output (summary),
 * and the execution data (the data payload passed to the execute_code API).
 * The tableData is now populated from executionData.
 */
function useLocalSqlData(
  sql?: string,
  executionOutput?: string,
  executionData?: any[]
) {
  const [chartData, setChartData] = useState<any[]>([]);
  const [tableData, setTableData] = useState<any[]>([]);
  const [dataFlow, setDataFlow] = useState<string>("");

  useEffect(() => {
    // If none of the parameters exist, do nothing.
    if (!sql && !executionOutput && !executionData) return;

    // Optionally, build chart data from SQL if needed.
    setChartData([]); // (Customize as necessary.)

    // Populate tableData with the executionData passed from the execute_code API.
    if (executionData) {
      setTableData(executionData);
    } else if (executionOutput) {
      // Fallback: Use execution output as a single-row table.
      setTableData([{ output: executionOutput }]);
    } else {
      setTableData([]);
    }

    // Build a simple data flow string.
    setDataFlow(`Data built from analysis payload: ${sql || ""}`);
  }, [sql, executionOutput, executionData]);

  return { chartData, tableData, dataFlow };
}

export function LLMResponseBubble({ message, position }: LLMResponseBubbleProps) {
  // Local state to control the active tab.
  const [tabValue, setTabValue] = useState(0);

  // Destructure content (the summary), sql (the generated code), and data (the execution payload data) from message.
  const { content, sql, data: executionData } = message as any;;

  // Use the updated hook to get chartData, tableData (from executionData), and dataFlow.
  const { chartData, tableData, dataFlow } = useLocalSqlData(
    sql,
    content,
    executionData
  );

  return (
    <Stack
      direction={position === "right" ? "row-reverse" : "row"}
      spacing={2}
      sx={{
        alignItems: "flex-start",
        minWidth: "100%",
        ml: position === "right" ? "auto" : 0,
        mr: position === "left" ? "auto" : 0,
      }}
    >
      <Card
        sx={{
          px: 2,
          py: 1,
          ...(position === "right" && {
            bgcolor: "var(--mui-palette-primary-main)",
            color: "var(--mui-palette-primary-contrastText)",
          }),
        }}
      >
        <Stack spacing={1}>
          {/* Display the execution output summary */}
          <Typography variant="body1">{content}</Typography>

          {/* Tabs for Chart, Data, SQL, and Data Flow */}
          <Tabs
            value={tabValue}
            onChange={(_, newVal) => setTabValue(newVal)}
            variant="fullWidth"
            sx={{ borderBottom: "1px solid var(--mui-palette-divider)" }}
          >
            <Tab label="Chart" />
            {/* <Tab label="Data" /> */}
            <Tab label="SQL" />
            {/* <Tab label="Data Flow" /> */}
          </Tabs>

          <Box sx={{ mt: 1 }}>
            {tabValue === 0 && <ChartTab chartOption={chartData} onChartRendered={function (): void {
              throw new Error("Function not implemented.");
            } } />}
            {/* {tabValue === 1 && <DataTab data={tableData} />} */}
            {tabValue === 2 && <SQLTab sql={sql || ""} />}
            {/* {tabValue === 3 && <DataFlowTab  />} */}
          </Box>
        </Stack>
      </Card>
    </Stack>
  );
}
