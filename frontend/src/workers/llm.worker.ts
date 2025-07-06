// src/workers/llm.worker.ts

import { ChatManager } from "../lib/web-llm-chat";

const chat = new ChatManager();


// Listen for messages from the React component
self.onmessage = async (event: MessageEvent) => {
  const { analysisData, parameterizedAnswer, originalQuestion } = event.data;

  // Construct a detailed prompt for the LLM
  const prompt = `
    You are an expert data analyst. Your task is to provide a clear, in-depth, natural language summary based on a data analysis that was just performed.

    Here is the user's original question:
    "${originalQuestion}"

    Here is a parameterized summary of the analysis results:
    "${parameterizedAnswer}"

    And here is the raw analysis data:
    ${JSON.stringify(analysisData, null, 2)}

    Based on all of this information, provide a comprehensive, easy-to-understand summary that directly answers the user's question. Do not start with "Certainly" or "Here is a summary". Just provide the summary directly.
    `;

  try {
    // Generate the summary, using the callback to post streaming messages back
    await chat.generate(prompt, (_step, message) => {
      // Post the streaming message chunk back to the main thread
      self.postMessage({
        type: "update",
        content: message,
      });
    });
  } catch (error: any) {
    self.postMessage({
      type: "error",
      content: error.message,
    });
  }
};