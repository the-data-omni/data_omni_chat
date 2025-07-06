import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  Stack,
  Typography,
} from '@mui/material';

// Define the models you support
const supportedModels = {
  openai: ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'],
  google: ['gemini-2.5-flash', 'gemini-2.5-pro'],
  anthropic: ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
  ollama: ['llama3:latest'],
};

type Provider = keyof typeof supportedModels;

export interface APISettings {
  provider: Provider;
  model: string;
  apiKey: string;
}

interface APISettingsModalProps {
  open: boolean;
  onClose: () => void;
  onSave: (settings: APISettings) => void;
  currentSettings: APISettings;
}

export function APISettingsModal({ open, onClose, onSave, currentSettings }: APISettingsModalProps) {
  const [provider, setProvider] = useState<Provider>(currentSettings.provider);
  const [model, setModel] = useState<string>(currentSettings.model);
  const [apiKey, setApiKey] = useState<string>(currentSettings.apiKey);
  const [ollamaModel, setOllamaModel] = useState(
    currentSettings.provider === 'ollama' ? currentSettings.model : 'llama3:latest'
  );

  const handleProviderChange = (event: SelectChangeEvent<string>) => {
    const newProvider = event.target.value as Provider;
    setProvider(newProvider);
    // Reset model to the first available for the new provider
        if (newProvider !== 'ollama') {
      setModel(supportedModels[newProvider][0]);
    }
  };

  const handleSave = () => {
    const finalModel = provider === 'ollama' ? ollamaModel : model;
    onSave({ provider, model: finalModel, apiKey });
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>API Settings</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 2 }}>
          <Typography variant="body2">
            API keys are stored only in your browser session and are sent securely with each request. We do not store them on our servers.
          </Typography>
          <FormControl fullWidth>
            <InputLabel id="provider-select-label">Provider</InputLabel>
            <Select
              labelId="provider-select-label"
              value={provider}
              label="Provider"
              onChange={handleProviderChange}
            >
              <MenuItem value="openai">OpenAI (GPT)</MenuItem>
              <MenuItem value="google">Google (Gemini)</MenuItem>
              <MenuItem value="anthropic">Anthropic (Claude)</MenuItem>
              <MenuItem value="ollama">Ollama (Local)</MenuItem>
            </Select>
          </FormControl>
          {/* --- CONDITIONALLY SHOW MODEL SELECTOR OR TEXT FIELD --- */}
          {provider === 'ollama' ? (
            <TextField
              fullWidth
              label="Ollama Model Name"
              value={ollamaModel}
              onChange={(e) => setOllamaModel(e.target.value)}
              placeholder="e.g., llama3, codellama:7b"
              helperText="Enter the name of a model installed on your local Ollama server."
              variant="outlined"
            />
          ) : (
            <FormControl fullWidth>
              <InputLabel id="model-select-label">Model</InputLabel>
              <Select
                labelId="model-select-label"
                value={model}
                label="Model"
                onChange={(e) => setModel(e.target.value)}
              >
                {supportedModels[provider].map((modelName) => (
                  <MenuItem key={modelName} value={modelName}>
                    {modelName}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}

          {/* --- HIDE API KEY FIELD FOR OLLAMA --- */}
          {provider !== 'ollama' && (
             <TextField
              fullWidth
              label="API Key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={`Enter your ${provider} API key`}
              variant="outlined"
            />
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained" disabled={provider !== 'ollama' && !apiKey}>
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
}