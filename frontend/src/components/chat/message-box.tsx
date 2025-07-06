"use client";

import * as React from "react";
import { useRef } from "react"; // Import useRef
import Avatar from "@mui/material/Avatar";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardMedia from "@mui/material/CardMedia";
import Link from "@mui/material/Link";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { Tabs, Tab, Button, CircularProgress } from "@mui/material";
import type { ECharts } from "echarts";

// --- NEW: Import the modern WebLLM API ---
import { CreateMLCEngine, MLCEngine } from "@mlc-ai/web-llm";

import { dayjs } from "@/lib/dayjs";
import type { Message } from "./types";
import { ChatContext } from "./chat-context";
import { getFileFromIndexedDB } from "@/hooks/utils/local_db";
import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm';
// eslint-disable-next-line import/default


// Tab components
import { ChartTab } from "@/components/dashboard/ChartTab";
import { SQLTab } from "@/components/dashboard/SQLTab";


const user = {
  id: "USR-000",
  name: "Sofia Rivers",
  avatar: "/assets/avatar.png",
} as const;

export interface MessageBoxProps {
  message: Message;
  onChartRendered: (chartInstance: ECharts) => void;
}

function LLMResponseBubble({ message, onChartRendered }: { message: Message; onChartRendered: (chartInstance: ECharts) => void; }) {
  // const { messages, updateMessage, activeFileName } = React.useContext(ChatContext);
  const { messages, updateMessage, activeFileName, analysisWorker } = React.useContext(ChatContext);

  const [isLoadingRealData, setIsLoadingRealData] = React.useState(false);
  const [loadingStatus, setLoadingStatus] = React.useState('');
  const [tabValue, setTabValue] = React.useState(0);
  
  
  // --- NEW: Use useRef to store the LLM engine instance ---
  const llmEngine = useRef<MLCEngine | null>(null);

  const { content, sql = "", chartData = [], profile, profileSynth } = message;
  const isProfileBubble = !!profile;
  const defaultTabsOpen = isProfileBubble ? false : chartData?.length > 0;
  const [tabsOpen, setTabsOpen] = React.useState(defaultTabsOpen);

  async function handleToggleRealData() {
    // 1. Check for required data from context and props
    if (!sql || !activeFileName || !analysisWorker) {
      console.error("Cannot run analysis: Missing generated code, active file name, or the shared worker.");
      alert("Cannot run analysis: App context is not ready.");
      return;
    }

    // 2. Find the original user question that prompted this LLM response
    const currentThreadMessages = messages.get(message.threadId) || [];
    const messageIndex = currentThreadMessages.findIndex(m => m.id === message.id);
    const originalQuestion = messageIndex > 0 ? currentThreadMessages[messageIndex - 1].content : "What does this data show?";

    // 3. Set initial loading states
    setIsLoadingRealData(true);
    setLoadingStatus("Analyzing real data...");
    updateMessage(message.id, { content: "Analyzing real data..." });

    try {
      // 4. Retrieve the original file from the browser's database
      const originalFile = await getFileFromIndexedDB(activeFileName);

      // 5. Define the handler for this specific request's response
      const handleAnalysisResult = async (event: MessageEvent) => {
        // Since the worker is shared, we must check if this message is for us.
        // The response from 'executeCode' is expected to have a 'result' key.
        if ('result' in event.data) {
          // Immediately remove the listener so it doesn't fire again for other messages
          analysisWorker.removeEventListener('message', handleAnalysisResult);

          const { status, result, error } = event.data;

          if (status === "complete") {
            // --- Analysis successful, now start the LLM summary generation ---
            updateMessage(message.id, { chartData: result.chart_data ?? [] });
            setLoadingStatus("Initializing AI Engine...");

            if (!llmEngine.current) {
              llmEngine.current = await CreateMLCEngine(
                "gemma-2b-it-q4f16_1-MLC",
                { initProgressCallback: (progress) => {
                    setLoadingStatus(`Loading AI: ${progress.text.replace('[...]', '')}`);
                  }
                }
              );
            }
            
            setLoadingStatus("Generating summary...");
            const prompt = `You are an expert data analyst. Your task is to provide a clear, in-depth, natural language summary based on a data analysis that was just performed.
            Here is the user's original question: "${originalQuestion}"
            Here is a parameterized summary of the analysis results: "${result.parameterized_answer}"
            And here is the raw analysis data: ${JSON.stringify(result.analysis_data, null, 2)}
            Based on all of this information, provide a comprehensive, easy-to-understand summary that directly answers the user's question. Do not start with "Certainly" or "Here is a summary". Just provide the summary directly.`;

            const llmMessages = [{ role: "user", content: prompt }];
            const chunks = await llmEngine.current.chat.completions.create({ stream: true, messages: llmMessages as ChatCompletionMessageParam[]  });

            let llmSummary = "";
            updateMessage(message.id, { content: "" }); // Clear previous content before streaming

            for await (const chunk of chunks) {
              const delta = chunk.choices[0]?.delta?.content || "";
              llmSummary += delta;
              updateMessage(message.id, { content: llmSummary });
            }
            
            setIsLoadingRealData(false);
            setLoadingStatus('');

          } else {
            throw new Error(error || "Analysis failed with an unknown error.");
          }
        }
      };
      
      // Add error handling specifically for the worker listener
      const handleWorkerError = (err: ErrorEvent) => {
        analysisWorker.removeEventListener('message', handleAnalysisResult);
        analysisWorker.removeEventListener('error', handleWorkerError);
        throw err;
      };

      // 6. Add the listeners right before we send the message
      analysisWorker.addEventListener('message', handleAnalysisResult);
      analysisWorker.addEventListener('error', handleWorkerError);

      // 7. Post the message to the shared worker
      analysisWorker.postMessage({
        type: 'executeCode',
        file: originalFile,
        codeToExecute: sql,
      });

    } catch (err: any) {
      console.error("Error during real data analysis:", err);
      updateMessage(message.id, { content: `An error occurred: ${err.message}` });
      setIsLoadingRealData(false);
      setLoadingStatus('');
    }
  }

  // ... (The rest of the component's JSX is below, with one small change to the button text)
  return (
    <Box sx={{ alignItems: "left", display: "flex", flex: "0 0 auto" }}>
      <Stack direction={"row"} spacing={2} sx={{ alignItems: "flex-start", minWidth: "75%", mr: "auto" }}>
        <Avatar src={message.author.avatar} />
        <Stack spacing={1} sx={{ flex: "1 1 auto" }}>
          <Card sx={{ px: 2, py: 1 }}>
            <Stack spacing={1}>
              <div>
                <Link color="inherit" variant="subtitle2" sx={{ cursor: "pointer" }}>
                  {message.author.name}
                </Link>
              </div>
              <Typography variant="body1" color="inherit" sx={{ whiteSpace: 'pre-wrap' }}>
                {content}
              </Typography>
              {!isProfileBubble && (
                <Button variant="outlined" size="small" sx={{ alignSelf: "flex-end", mb: 1 }} onClick={handleToggleRealData} disabled={isLoadingRealData}>
                  {isLoadingRealData ? <CircularProgress size={20} sx={{mr: 1}} /> : null}
                  {isLoadingRealData ? loadingStatus : 'Toggle Real Data'}
                </Button>
              )}
              <Button variant="outlined" size="small" onClick={() => setTabsOpen((prev) => !prev)} sx={{ alignSelf: "flex-end", mb: 1 }}>
                {tabsOpen ? "Hide Details" : "Show Details"}
              </Button>
              {tabsOpen && (
                <>
                  <Tabs value={tabValue} onChange={(_, newVal) => setTabValue(newVal)} variant="fullWidth" sx={{ borderBottom: "1px solid var(--mui-palette-divider)" }}>
                    <Tab label="Chart" />
                    <Tab label="Code" />
                  </Tabs>
                  <Box sx={{ mt: 1 }}>
                    {tabValue === 0 &&  <ChartTab chartOption={chartData} onChartRendered={onChartRendered} />}
                    {tabValue === 1 && <SQLTab sql={sql || "No code available"} />}
                  </Box>
                </>
              )}
            </Stack>
          </Card>
          <Box sx={{ display: "flex", justifyContent: "flex-start", px: 2 }}>
            <Typography color="text.secondary" noWrap variant="caption">{dayjs(message.createdAt).fromNow()}</Typography>
          </Box>
        </Stack>
      </Stack>
    </Box>
  );
}


function DefaultBubble({ message }: { message: Message }) {
  const position = message.author.id === user.id ? "right" : "left";

  return (
    <Box
      sx={{
        alignItems: position === "right" ? "flex-end" : "flex-start",
        display: "flex",
        flex: "0 0 auto",
      }}
    >
      <Stack
        direction={position === "right" ? "row-reverse" : "row"}
        spacing={2}
        sx={{
          alignItems: "flex-start",
          maxWidth: "75%",
          ml: position === "right" ? "auto" : 0,
          mr: position === "left" ? "auto" : 0,
        }}
      >
        <Avatar src={message.author.avatar} />
        <Stack spacing={1} sx={{ flex: "1 1 auto" }}>
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
              <div>
                <Link color="inherit" variant="subtitle2" sx={{ cursor: "pointer" }}>
                  {message.author.name}
                </Link>
              </div>
              {message.type === "image" ? (
                <CardMedia image={message.content} sx={{ height: "200px", width: "200px" }} />
              ) : null}
              {message.type === "text" ? (
                <Typography variant="body1" color="inherit">
                  {message.content}
                </Typography>
              ) : null}
            </Stack>
          </Card>
          <Box
            sx={{
              display: "flex",
              justifyContent: position === "right" ? "flex-end" : "flex-start",
              px: 2,
            }}
          >
            <Typography color="text.secondary" noWrap variant="caption">
              {dayjs(message.createdAt).fromNow()}
            </Typography>
          </Box>
        </Stack>
      </Stack>
    </Box>
  );
}

export function MessageBox({ message, onChartRendered }: MessageBoxProps) {
  if (message.type === "llm") {
    return <LLMResponseBubble message={message} onChartRendered={onChartRendered} />;
  }
  return <DefaultBubble message={message} />;
}