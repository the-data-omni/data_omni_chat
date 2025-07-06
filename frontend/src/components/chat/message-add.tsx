"use client";

import * as React from "react";
import IconButton from "@mui/material/IconButton";
import OutlinedInput from "@mui/material/OutlinedInput";
import Stack from "@mui/material/Stack";
import Tooltip from "@mui/material/Tooltip";
import { Camera as CameraIcon } from "@phosphor-icons/react/dist/ssr/Camera";
import { Paperclip as PaperclipIcon } from "@phosphor-icons/react/dist/ssr/Paperclip";
import { PaperPlaneTilt as PaperPlaneTiltIcon } from "@phosphor-icons/react/dist/ssr/PaperPlaneTilt";
import Modal1 from "@/components/dashboard/modal-1";
import type { MessageType } from "./types";

export interface MessageAddProps {
  disabled?: boolean;
  onSend?: (type: MessageType, content: string) => void | Promise<void>;
}

export function MessageAdd({ disabled = false, onSend }: MessageAddProps): React.JSX.Element {
  const [content, setContent] = React.useState("");
  const [modalOpen, setModalOpen] = React.useState(false);

  const handleAttach = React.useCallback(() => {
    setModalOpen(true);
  }, []);

  const handleChange = React.useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setContent(event.target.value);
  }, []);

  const handleSend = React.useCallback(() => {
    if (!content) return;
    onSend?.("text", content);
    setContent("");
  }, [content, onSend]);

  const handleKeyUp = React.useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.code === "Enter") handleSend();
    },
    [handleSend]
  );

  return (
    <>
      <Stack direction="row" spacing={2} sx={{ alignItems: "center", flex: "0 0 auto", px: 3, py: 1 }}>
        <OutlinedInput
          disabled={disabled}
          onChange={handleChange}
          onKeyUp={handleKeyUp}
          // --- MODIFIED LINE: Conditional placeholder text ---
          placeholder={disabled ? "Please upload a file to begin" : "Leave a message"}
          sx={{ flex: "1 1 auto" }}
          value={content}
        />
        <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
          <Tooltip title="Send">
            <span>
              <IconButton
                color="primary"
                disabled={!content || disabled}
                onClick={handleSend}
                sx={{
                  bgcolor: "var(--mui-palette-primary-main)",
                  color: "var(--mui-palette-primary-contrastText)",
                  "&:hover": { bgcolor: "var(--mui-palette-primary-dark)" },
                }}
              >
                <PaperPlaneTiltIcon />
              </IconButton>
            </span>
          </Tooltip>
          <Stack direction="row" spacing={1} sx={{ display: { xs: "none", sm: "flex" } }}>
            <Tooltip title="Attach photo">
              <span>
                <IconButton disabled={disabled} edge="end" onClick={handleAttach}>
                  <CameraIcon />
                </IconButton>
              </span>
            </Tooltip>
            <Tooltip title="Attach file">
              <span>
                <IconButton disabled={disabled} edge="end" onClick={handleAttach}>
                  <PaperclipIcon />
                </IconButton>
              </span>
            </Tooltip>
          </Stack>
        </Stack>
      </Stack>
      {modalOpen && <Modal1 onClose={() => setModalOpen(false)} />}
    </>
  );
}