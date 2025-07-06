

"use client";

import React, { useCallback, useContext, useEffect, useRef, useState } from "react";
import { Avatar, Box, Button, CircularProgress, Stack, Typography } from "@mui/material";
import UploadFileIcon from '@mui/icons-material/UploadFile';
import type { ECharts } from "echarts";
import PptxGenJS from "pptxgenjs";
// import { fetchWithAuth } from "@/lib/apiClient"; // Adjust the import path as needed

import Modal1 from "@/components/dashboard/modal-1";
import { ChatContext } from "./chat-context";
import { MessageAdd } from "./message-add";
import { MessageBox } from "./message-box";
import { ThreadToolbar } from "./thread-toolbar";
import type { APISettings } from '../dashboard/api-settings-modal'; // Import the type
import type { MessageType, ThreadType } from "./types";

async function fetchJSON<T>(url: string): Promise<T> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GET ${url} → ${res.status}`);
    return res.json() as Promise<T>;
}

function LoadingMessage() {
    return (
        <Stack direction="row" spacing={2} sx={{ justifyContent: "flex-start" }}>
            <Avatar sx={{ bgcolor: "primary.main", color: "white" }}>C</Avatar>
            <Box sx={{ bgcolor: "grey.100", borderRadius: "10px", borderTopLeftRadius: "0px", p: "10px 15px", display: "flex", alignItems: "center", gap: 2 }}>
                <CircularProgress size={22} />
                <Typography variant="body1" sx={{ color: "text.secondary" }}>Processing...</Typography>
            </Box>
        </Stack>
    );
}

interface UploadPromptProps {
  onAttach: () => void;
}
function UploadPrompt({ onAttach }: UploadPromptProps) {
    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', textAlign: 'center', color: 'text.secondary', p: 3 }}>
            <UploadFileIcon sx={{ fontSize: '52px', mb: 2 }} />
            <Typography variant="h6" component="p" gutterBottom>No Data Loaded</Typography>
            <Typography variant="body1" sx={{ mb: 3 }}>Please upload a CSV file to begin your analysis.</Typography>
            <Button variant="contained" onClick={onAttach}>Attach File</Button>
        </Box>
    );
}

interface AIThreadViewProps {
    threadId: string;
    threadType: ThreadType;
}

type ConnectionStatus = 'unknown' | 'verifying' | 'success' | 'error';

export function AIThreadView({ threadId }: AIThreadViewProps) {

    const { createMessage, markAsRead, threads, messages } = useContext(ChatContext);
    const messagesRef = useRef<HTMLDivElement>(null);

    const [originalData, setOriginalData] = useState<unknown[]>([]);
    const [chartInstances, setChartInstances] = useState<Map<string, ECharts>>(new Map());
    const [isProcessing, setIsProcessing] = useState<boolean>(false);
    const [isDataLoading, setIsDataLoading] = useState<boolean>(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [chatIsActive, setChatIsActive] = useState(false);

    const [conversationId, setConversationId] = useState<string | null>(null);

    const [apiSettings, setApiSettings] = useState<APISettings>({
        provider: 'openai',
        model: 'gpt-4o',
        apiKey: '', // Initially empty
    });


    const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('unknown');
    
    // --- NEW FUNCTION TO VERIFY CONNECTION ---
    const verifyApiConnection = useCallback(async (settings: APISettings) => {
        // Ollama doesn't require a key, we can verify immediately
        if (settings.provider === 'ollama') {
             // A more robust check would still hit the backend to see if it can reach localhost
        } else if (!settings.apiKey) {
            setConnectionStatus('unknown'); // Reset if key is removed
            return;
        }

        setConnectionStatus('verifying');
        try {
            const res = await fetch("http://127.0.0.1:8000/verify_connection", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${settings.apiKey}`
                },
                body: JSON.stringify({ model: settings.model })
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Connection failed.');
            }
            
            setConnectionStatus('success');

        } catch (error: any) {
            console.error("Connection verification failed:", error);
            setConnectionStatus('error');
            // Optionally, show an alert or snackbar with the error message
            alert(`Connection failed: ${error.message}`);
        }
    }, []);

    const reloadData = useCallback(() => {
        setIsDataLoading(true);
        fetchJSON<unknown[]>("http://127.0.0.1:8000/data/original")
            .then(setOriginalData)
            .catch(() => setOriginalData([]))
            .finally(() => setIsDataLoading(false));
    }, []);

    const handleModalOpen = useCallback(() => { setIsModalOpen(true); }, []);
    
    const handleModalClose = useCallback((syntheticDataGenerated: boolean) => {
        setIsModalOpen(false);
        if (syntheticDataGenerated) {
            setChatIsActive(true);
            reloadData();
            setConversationId(null);
        }
    }, [reloadData]);

    const handleChartRendered = useCallback((messageId: string, chartInstance: ECharts) => {
        setChartInstances((prev) => new Map(prev).set(messageId, chartInstance));
    }, []);

    const handleSendMessage = useCallback(async (type: MessageType, content: string) => {
        if (!apiSettings.apiKey) {
            alert("Please set your API key in the settings before sending a message.");
            createMessage({ threadId, type: 'llm', content: "Error: API Key not set." });
            return;
        }
        createMessage({ threadId, type, content });
        setIsProcessing(true);
        try {
            const res = await fetch("http://127.0.0.1:8000/full_analysis", {
                method: "POST",
                headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiSettings.apiKey}` },
                body: JSON.stringify({ data: originalData, question: content,conversation_id: conversationId, model: apiSettings.model}),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const {
                generated_code: sql = "",
                summary = "No summary",
                chart_data: chartData = null,
                parameterized_summary: parameterizedSummary = "No summary",
                conversation_id: newConversationId,
            } = await res.json();

            if (newConversationId) {
                setConversationId(newConversationId);
            }
            createMessage({ threadId, type: "llm", content: parameterizedSummary || summary, sql, chartData });
        } catch (error: any) {
            createMessage({ threadId, type: "llm", content: `Error: ${error.message}`, sql: "" });
        } finally {
            setIsProcessing(false);
        }
    }, [threadId, createMessage, originalData, conversationId, apiSettings]);

    useEffect(() => markAsRead(threadId), [threadId, markAsRead]);

    useEffect(() => {
        if (messagesRef.current) {
            messagesRef.current.scrollTo({ top: messagesRef.current.scrollHeight, behavior: "smooth" });
        }
    }, [messages, threadId, isProcessing]);

    const thread = threads.find((t) => t.id === threadId);
    
    const handleSettingsSave = useCallback((newSettings: APISettings) => {
        setApiSettings(newSettings);
        // --- TRIGGER VERIFICATION ON SAVE ---
        verifyApiConnection(newSettings);
    }, [verifyApiConnection]);
	
const handleExportToGoogleSlides = useCallback(async () => {
    const threadMessages = messages.get(threadId) ?? [];
    if (threadMessages.length === 0) {
        alert("No content to export.");
        return;
    }

    //First we want to generate the ppt in memory ---
    const pptx = new PptxGenJS();
    pptx.layout = 'LAYOUT_WIDE';
    const reportTitle = threads.find(t => t.id === threadId)?.participants[0]?.name || "Analysis Report";

    pptx.addSlide().addText(`Analysis Report: ${reportTitle}`, { x: 0.5, y: 2.5, w: '90%', h: 1.5, fontSize: 36, bold: true, align: 'center' });

    for (const msg of threadMessages) {
        if (msg.type !== 'llm' || (!msg.content && !msg.chartData)) continue;

        const chartInstance = chartInstances.get(msg.id);
        let slideTitle = 'Analysis Slide';
        if (chartInstance) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const titleOption = chartInstance.getOption()?.title as any;
            slideTitle = titleOption?.text || slideTitle;
        }

        const slide = pptx.addSlide();
        slide.addText(slideTitle, { x: 0.5, y: 0.2, w: '90%', h: 0.5, fontSize: 24, bold: true, align: 'center' });
        
        if (chartInstance) {
            const chartImageBase64 = chartInstance.getDataURL({ type: 'png', pixelRatio: 3, backgroundColor: '#fff' });
            slide.addImage({ data: chartImageBase64, x: 0.5, y: 0.8, w: 12.33, h: 5 });
        }
        
        const summaryText = msg.parameterizedSummary || msg.content || 'No summary available.';
        slide.addText(summaryText, { x: 0.5, y: 6, w: 12.33, h: 1.2, fontSize: 14, align: 'center' });
    }

    try {
        // --- 2. Get the generated PPTX as a Blob ---
        const pptxBlob = await pptx.write({ outputType: 'blob' }) as Blob;

        // --- 3. Prepare FormData to send the file ---
        const formData = new FormData();
        formData.append('pptx_file', pptxBlob, `${reportTitle}.pptx`);

        // --- 4. Send the file to the new backend endpoint ---
        const res = await fetch("http://127.0.0.1:8000/upload/convert-pptx-to-slides", {
            method: "POST",
            body: formData, // The body is now FormData, not a JSON string
        });

        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server responded with status: ${res.status}`);
        }

        const responseData = await res.json();
        window.open(responseData.presentationUrl, '_blank');
        alert("Google Slides presentation created successfully!");

    } catch (error: any) {
        console.error("Failed to create Google Slides from PPTX:", error);
        alert(`An error occurred during conversion: ${error.message}`);
    }
}, [messages, threadId, threads, chartInstances]);

    const handleExportToPowerPoint = useCallback(async () => {
        const threadMessages = messages.get(threadId) ?? [];
        if (threadMessages.length === 0) {
            alert("No content to export.");
            return;
        }
        const reportTitle = thread?.participants[0]?.name || "Analysis";
        const pptx = new PptxGenJS();
        pptx.layout = 'LAYOUT_WIDE';
        pptx.addSlide().addText(`Analysis Report: ${reportTitle}`, { x: 0.5, y: 2.5, w: '90%', h: 1.5, fontSize: 36, bold: true, align: 'center' });
        for (const msg of threadMessages) {
            if (msg.type !== 'llm' || (!msg.content && !msg.chartData)) continue;
            const chartInstance = chartInstances.get(msg.id);
            let slideTitle = `Analysis Slide`;
            if (chartInstance) {
                const chartOptions = chartInstance.getOption();
                const titleOption = chartOptions?.title;
                if (Array.isArray(titleOption) && titleOption.length > 0 && titleOption[0].text) {
                    slideTitle = titleOption[0].text as string;
			} else if (titleOption && typeof titleOption === 'object' && !Array.isArray(titleOption) && 'text' in titleOption) {
				// TypeScript now knows titleOption has a 'text' property here
				slideTitle = titleOption.text as string; 
			}
            }
            const slide = pptx.addSlide();
            slide.addText(slideTitle, { x: 0.5, y: 0.2, w: 12.33, h: 0.5, fontSize: 24, bold: true, align: 'center' });
            if (chartInstance) {
                const chartImageBase64 = chartInstance.getDataURL({ type: 'png', pixelRatio: 3, backgroundColor: '#fff' });
                slide.addImage({ data: chartImageBase64, x: 0.5, y: 0.8, w: 12.33, h: 5 });
            }
            const summaryText = msg.parameterizedSummary || msg.content || 'No summary available.';
            slide.addText(summaryText, { x: 0.5, y: 6, w: 12.33, h: 1.2, fontSize: 14, align: 'center' });
        }
        await pptx.writeFile({ fileName: `Report - ${reportTitle}.pptx` });
    }, [messages, threadId, thread, chartInstances]);

    const threadMessages = messages.get(threadId) ?? [];
    if (!thread) {
        return (
            <Box sx={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Typography color="textSecondary" variant="h6">Thread not found</Typography>
            </Box>
        );
    }

    const showUploadPrompt = !chatIsActive && threadMessages.length === 0;
    const isChatDisabled = !chatIsActive || isProcessing || isDataLoading;

    return (
        <Box sx={{ display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" }}>
            {isModalOpen && <Modal1 onClose={handleModalClose} />}
            <Box sx={{ flex: "0 0 auto", borderBottom: "1px solid #ccc" }}>
                <ThreadToolbar
                    thread={thread}
                    onExport={handleExportToPowerPoint}
                    onExportToGoogleSlides={handleExportToGoogleSlides}
                    apiSettings={apiSettings}
                    onSettingsSave={handleSettingsSave}
                    connectionStatus={connectionStatus}
                    modelName={apiSettings.model}
                />
            </Box>
            <Box ref={messagesRef} sx={{ flex: "1 1 auto", overflowY: "auto", p: 2 }}>
                {isDataLoading ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                        <CircularProgress />
                    </Box>
                ) : showUploadPrompt ? (
                    <UploadPrompt onAttach={handleModalOpen} />
                ) : (
                    <Stack spacing={2}>
                        {threadMessages.map((msg) => (
                            <MessageBox
                                key={msg.id}
                                message={msg}
                                onChartRendered={(chartInstance) => handleChartRendered(msg.id, chartInstance)}
                            />
                        ))}
                        {isProcessing && <LoadingMessage />}
                    </Stack>
                )}
            </Box>
            <Box sx={{ flex: "0 0 auto", borderTop: "1px solid #ccc", p: 2 }}>
                <MessageAdd onSend={handleSendMessage} disabled={isChatDisabled} />
            </Box>
        </Box>
    );
}