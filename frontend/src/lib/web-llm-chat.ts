// // // src/lib/web-llm-chat.ts

// // import * as webllm from "@mlc-ai/web-llm";

// // export class ChatManager {
// //   private chat: webllm.WebLLM;
// //   private appConfig: webllm.AppConfig = {
// //     chat_opts: {
// //       temperature: 0.7,
// //       top_p: 0.95,
// //       // The WebLLM model complains if `repetition_penalty` is not set.
// //       repetition_penalty: 1,
// //     },
// //   };

// //   constructor() {
// //     // eslint-disable-next-line import/namespace
// //     this.chat = new webllm.ChatModule();
// //   }

// //   // Use a callback to stream partial responses back to the main thread
// //   async generate(
// //     prompt: string,
// //     progressCallback: (step: number, message: string) => void
// //   ) {
// //     await this.chat.reload(
// //       "gemma-2b-it-q4f16_1-MLC", // Using Gemma 2B with 16-bit float quantization
// //       this.appConfig,
// //     );
    
// //     try {
// //         await this.chat.generate(prompt, progressCallback);
// //     } catch (error: any) {
// //         console.error("Error during chat generation:", error);
// //         // Relay the error message back
// //         progressCallback(-1, error.message);
// //     }
// //   }
// // }
// import * as webllm from "@mlc-ai/web-llm";

// export class ChatManager {
//   /** One local engine instance – think “model session”. */
//   private engine: webllm.MLCEngine;

//   constructor(
//     private readonly initProgress?: (p: number) => void,
//   ) {
//     // NB: init is synchronous; model download happens in `reload`
//     this.engine = new webllm.MLCEngine({
//       initProgressCallback: this.initProgress,
//     });
//   }

//   /** Stream tokens back to the caller. */
//   async generate(
//     prompt: string,
//     onDelta: (step: number, fragment: string) => void,
//   ) {
//     /* 1. Ensure the model is loaded (first call downloads & caches). */
//     await this.engine.reload(
//       "gemma-2b-it-q4f16_1-MLC",
//       /* ChatOptions */ { temperature: 0.7, top_p: 0.95, repetition_penalty: 1 },
//     );

//     /* 2. Run the chat completion (stream = true gives AsyncIterable). */
//     const chunks = await this.engine.chat.completions.create({
//       messages: [{ role: "user", content: prompt }],
//       stream: true,
//       temperature: 0.7,
//       top_p: 0.95,
//       // repetition_penalty: 1,
//     });

//     /* 3. Deliver partial tokens. */
//     let step = 0;
//     for await (const chunk of chunks) {
//       const delta = chunk.choices[0]?.delta.content ?? "";
//       if (delta) onDelta(step++, delta);
//     }
//   }
// }
import * as webllm from "@mlc-ai/web-llm";

export class ChatManager {
  /** Underlying WebLLM engine (“model session”). */
  private engine: webllm.MLCEngine;

  /**
   * @param initProgress Optional callback receiving a numeric progress (0–1 or 0–100)
   *                     as the model downloads.
   */
  constructor(
    private readonly initProgress?: (progress: number) => void,
  ) {
    this.engine = new webllm.MLCEngine({
      initProgressCallback: initProgress
        ? (report: webllm.InitProgressReport) => {
            // Most examples use a single numeric field on the report
            const p = (report as any).progress;
            if (typeof p === "number") {
              this.initProgress!(p);
            }
          }
        : undefined,
    });
  }

  /**
   * Streams a chat completion for `prompt`, invoking `onDelta` for each token chunk.
   */
  async generate(
    prompt: string,
    onDelta: (step: number, fragment: string) => void,
  ) {
    // 1. Load (or reload) the model with desired logit control
    await this.engine.reload(
      "gemma-2b-it-q4f16_1-MLC",
      { temperature: 0.7, top_p: 0.95, repetition_penalty: 1 },
    );

    // 2. Kick off a streaming chat completion (only temperature & top_p are allowed)
    const chunks = await this.engine.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      stream: true,
      temperature: 0.7,
      top_p: 0.95,
    });

    // 3. Deliver each partial content piece
    let step = 0;
    for await (const chunk of chunks) {
      const delta = chunk.choices[0]?.delta.content ?? "";
      if (delta) onDelta(step++, delta);
    }
  }
}
