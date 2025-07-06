export interface Contact {
	id: string;
	name: string;
	avatar?: string;
	isActive: boolean;
	lastActivity?: Date;
}

export type ThreadType = "direct" | "group";

export interface Thread {
	id: string;
	type: ThreadType;
	participants: Participant[];
	unreadCount: number;
}

export interface Participant {
	id: string;
	name: string;
	avatar?: string;
}

export type MessageType = "text" | "image" | "llm";


export interface Message {
  profileSynth: any;
  id: string;
  threadId: string;
  type: "text" | "image" | "llm";
  content: string;
  sql?: string;
  chartData?: any[]; // <-- add this line
  createdAt: Date;
  profile?:any;
  generated_code?: string;
  parameterizedSummary?: string;
  author: {
    id: string;
    name: string;
    avatar: string;
  };
}
