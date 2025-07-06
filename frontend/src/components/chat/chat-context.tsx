
"use client";

import React, { createContext, useState, useCallback, useRef, useEffect } from "react";
import { v4 as uuidv4 } from 'uuid'; // Standard for unique IDs
import type { Contact, Message, MessageType, Thread } from "./types";
// eslint-disable-next-line import/default
import AnalysisWorker from '@/workers/anonymizer.worker.ts?worker';

// A simple no-op function for defaults
function noop(): void {}

export type CreateThreadParams =
  | { type: "direct"; recipientId: string }
  | { type: "group"; recipientIds: string[] };

// This is the data needed to CREATE a message. The final Message type will have more fields.
export interface CreateMessageParams {
  threadId: string;
  type: MessageType; // "text" | "image" | "llm"
  content: string;
  chartData?: any[];
  sql?: string;
  profile?: any;
  profileSynth?: any;
  data?: {}; // optional
  generatedCode?: string;
  parameterizedSummary?: string;
}

export interface ChatContextValue {
  contacts: Contact[];
  threads: Thread[];
  messages: Map<string, Message[]>;
  createThread: (params: CreateThreadParams) => string;
  markAsRead: (threadId: string) => void;
  createMessage: (params: CreateMessageParams) => void;
  updateMessage: (messageId: string, updates: Partial<Message>) => void; // Changed from 'changes' to 'updates' for clarity
  openDesktopSidebar: boolean;
  setOpenDesktopSidebar: React.Dispatch<React.SetStateAction<boolean>>;
  openMobileSidebar: boolean;
  setOpenMobileSidebar: React.Dispatch<React.SetStateAction<boolean>>;
  activeFileName: string | null;
  setActiveFileName: (name: string | null) => void;
  analysisWorker: Worker | null;
}

export const ChatContext = createContext<ChatContextValue>({
  contacts: [],
  threads: [],
  messages: new Map(),
  createThread: noop as () => string,
  markAsRead: noop,
  createMessage: noop,
  updateMessage: noop,
  openDesktopSidebar: true,
  setOpenDesktopSidebar: noop,
  openMobileSidebar: true,
  setOpenMobileSidebar: noop,
  activeFileName: null,
  setActiveFileName: noop,
  analysisWorker: null,
});

export interface ChatProviderProps {
  children: React.ReactNode;
  contacts: Contact[];
  threads: Thread[];
  messages: Message[];
}

export function ChatProvider({
  children,
  contacts: initialContacts = [],
  threads: initialThreads = [],
  messages: initialMessages = [],
}: ChatProviderProps): React.JSX.Element {
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [messageMap, setMessageMap] = useState<Map<string, Message[]>>(new Map());

  const [openDesktopSidebar, setOpenDesktopSidebar] = useState<boolean>(true);
  const [openMobileSidebar, setOpenMobileSidebar] = useState<boolean>(false);
  const [activeFileName, setActiveFileName] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);

  // Initialize and manage the shared web worker's lifecycle
  useEffect(() => {
    if (workerRef.current === null) {
      console.log("ChatProvider: Creating the shared analysis worker.");
      workerRef.current = new AnalysisWorker();
      workerRef.current.postMessage({ type: 'init' });

      workerRef.current.onerror = (event) => {
        console.error("A critical error occurred in the analysis worker:", event);
      };
      
      workerRef.current.onmessage = (event) => {
        if (event.data.status === 'ready') {
          console.log("ChatProvider: Worker has confirmed Pyodide is ready.");
        }
      };
    }

    const worker = workerRef.current;
    return () => {
      if (worker) {
        console.log("ChatProvider: Terminating the analysis worker.");
        worker.terminate();
        workerRef.current = null;
      }
    };
  }, []);

  // Sync initial props to state
  useEffect(() => { setContacts(initialContacts); }, [initialContacts]);
  useEffect(() => { setThreads(initialThreads); }, [initialThreads]);
  useEffect(() => {
    const newMap = new Map<string, Message[]>();
    initialMessages.forEach((msg) => {
      const thread = newMap.get(msg.threadId) || [];
      thread.push(msg);
      newMap.set(msg.threadId, thread);
    });
    setMessageMap(newMap);
  }, [initialMessages]);

  const createThread = useCallback((params: CreateThreadParams): string => {
    // Placeholder for thread creation logic
    return "THREAD_ID";
  }, []);

  const markAsRead = useCallback((threadId: string) => {
    // Placeholder for marking a thread as read
  }, []);

  /**
   * IMMUTABLE IMPLEMENTATION
   * Add a new message to a thread.
   */
  const createMessage = useCallback((params: CreateMessageParams): void => {
    const author = params.type === "llm"
      ? { id: "LLM-123", name: "The Data Omni Chat", avatar: "/favicon.ico" }
      : { id: "USR-000", name: "Me", avatar: "/robot-avatar.png" };

    const newMessage: Message = {
      id: uuidv4(), // Generate a new, truly unique ID
      threadId: params.threadId,
      type: params.type,
      author,
      content: params.content,
      createdAt: new Date(),
      sql: params.sql,
      chartData: params.chartData ?? [],
      profile: params.profile,
      profileSynth: params.profileSynth,
      generated_code: params.sql || "",
      parameterizedSummary: params.parameterizedSummary || "",
    };

    setMessageMap((prevMap) => {
      const newMap = new Map(prevMap); // 1. Create a new Map
      const thread = newMap.get(params.threadId) || []; // 2. Get the current thread
      const newThread = [...thread, newMessage]; // 3. Create a new array with the new message
      newMap.set(params.threadId, newThread); // 4. Set the new array in the new map
      return newMap; // 5. Return the new map to update state
    });
  }, []); 

  /**
   * IMMUTABLE IMPLEMENTATION
   * Update an existing message by ID with partial changes.
   */
  const updateMessage = useCallback((messageId: string, updates: Partial<Message>) => {
    setMessageMap((prevMap) => {
      let targetThreadId: string | undefined;

      // Find which thread the message belongs to
      for (const [threadId, messages] of prevMap.entries()) {
        if (messages.some(msg => msg.id === messageId)) {
          targetThreadId = threadId;
          break;
        }
      }

      if (!targetThreadId) return prevMap; // Return original map if message not found

      const newMap = new Map(prevMap); // 1. Create a new Map
      const thread = newMap.get(targetThreadId)!; // We know the thread exists

      // 2. Create a new array by mapping over the old one
      const newThread = thread.map(msg => 
        msg.id === messageId 
          ? { ...msg, ...updates } // 3. If it's our message, create a new object with updates
          : msg // Otherwise, return the original object
      );
      
      newMap.set(targetThreadId, newThread); // 4. Set the new array in the new map
      return newMap; // 5. Return the new map to update state
    });
  }, []);

  const value: ChatContextValue = {
    contacts,
    threads,
    messages: messageMap,
    createThread,
    markAsRead,
    createMessage,
    updateMessage,
    openDesktopSidebar,
    setOpenDesktopSidebar,
    openMobileSidebar,
    setOpenMobileSidebar,
    activeFileName,
    setActiveFileName,
    analysisWorker: workerRef.current,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}