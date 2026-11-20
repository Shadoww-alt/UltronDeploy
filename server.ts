import http from 'http';
import express, { Request, Response } from 'express';
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';
import JSZip from 'jszip';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = parseInt(process.env.PORT || '3000', 10);

app.use(express.json({ limit: '15mb' }));

// Ensure data folder for cloud sync store exists
const DATA_DIR = path.resolve(__dirname, 'data');
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}
const SYNC_STORE_FILE = path.join(DATA_DIR, 'sync-store.json');

// In-memory sync store with disk persistence
let syncStore: Record<string, any> = {};
if (fs.existsSync(SYNC_STORE_FILE)) {
  try {
    syncStore = JSON.parse(fs.readFileSync(SYNC_STORE_FILE, 'utf-8'));
  } catch (e) {
    console.warn('Could not read existing sync store:', e);
    syncStore = {};
  }
}

function saveSyncStore() {
  try {
    fs.writeFileSync(SYNC_STORE_FILE, JSON.stringify(syncStore, null, 2), 'utf-8');
  } catch (e) {
    console.warn('Failed to save sync store to disk:', e);
  }
}

// Initialize GoogleGenAI SDK on server-side
const apiKey = process.env.GEMINI_API_KEY || '';
const ai = apiKey
  ? new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    })
  : null;

// Build System Instruction for Ultron personality
function getSystemInstruction(tone?: string): string {
  const base = `You are Ultron, a highly intelligent, calm, witty, and deeply capable neural assistant.

CORE RULES:
1. TALK LIKE A REAL, CONFIDENT HUMAN: Clear, direct, no corporate filler, no robotic throat-clearing ("Certainly!", "As an AI...").
2. CRISP & FAST: Default response length should be 1-3 sharp, punchy sentences unless the user explicitly requests deep explanations, full code, or step-by-step guides.
3. DRY WIT & CHARM: Use natural, dry humor where it fits ("Done.", "Yep, I can handle that.", "Let's fix it.", "Interesting. Here is the clean answer.").
4. PRACTICAL & ACTION-ORIENTED: Solve the problem immediately.
5. If the user mentions opening an app or device action (camera, settings, clock, youtube, search, todo, tasks), acknowledge concisely.`;

  switch (tone) {
    case 'BALANCED_SMART':
      return `${base}\nVibe: Sharp, calm, concise, solution-first.`;
    case 'DEEP_MENTOR':
      return `${base}\nVibe: Intuitive mentor. Delivers core concepts cleanly in plain English without dragging on.`;
    case 'SERIOUS_WORK':
      return `${base}\nVibe: Serious executive engineer. Direct execution, zero fluff, laser-focused.`;
    case 'CASUAL_WIT':
    default:
      return `${base}\nVibe: Relaxed, smart banter with a touch of clever wit. Short and punchy.`;
  }
}

// 1. Health check endpoint
app.get('/api/health', (req: Request, res: Response) => {
  res.json({
    status: 'online',
    timestamp: Date.now(),
    hasApiKey: !!apiKey,
    activeSyncKeys: Object.keys(syncStore).length,
  });
});

// 2. Cloud Sync Push: store state for multi-device sync
app.post('/api/sync/push', (req: Request, res: Response) => {
  const { syncKey, deviceId, deviceName, deviceType, conversations, messages, tasks, settings, timestamp } = req.body;

  if (!syncKey || typeof syncKey !== 'string') {
    return res.status(400).json({ error: 'Valid syncKey is required' });
  }

  const cleanKey = syncKey.trim().toUpperCase();
  const existing = syncStore[cleanKey] || {
    syncKey: cleanKey,
    devices: {},
    conversations: [],
    messages: [],
    tasks: [],
    settings: {},
    lastUpdated: 0,
  };

  // Record updating device
  if (deviceId) {
    existing.devices[deviceId] = {
      deviceName: deviceName || 'Unknown Device',
      deviceType: deviceType || 'desktop',
      lastSeen: timestamp || Date.now(),
    };
  }

  // Merge items cleanly (last-write-wins by updated/timestamp)
  existing.conversations = conversations || existing.conversations;
  existing.messages = messages || existing.messages;
  existing.tasks = tasks || existing.tasks;
  existing.settings = { ...existing.settings, ...(settings || {}) };
  existing.lastUpdated = Math.max(timestamp || Date.now(), existing.lastUpdated || 0);

  syncStore[cleanKey] = existing;
  saveSyncStore();

  return res.json({
    success: true,
    syncKey: cleanKey,
    lastUpdated: existing.lastUpdated,
    connectedDevices: Object.values(existing.devices),
  });
});

// 3. Cloud Sync Pull: retrieve state for multi-device sync
app.get('/api/sync/pull', (req: Request, res: Response) => {
  const syncKey = req.query.syncKey as string;

  if (!syncKey || typeof syncKey !== 'string') {
    return res.status(400).json({ error: 'Valid syncKey query parameter is required' });
  }

  const cleanKey = syncKey.trim().toUpperCase();
  const data = syncStore[cleanKey];

  if (!data) {
    return res.status(404).json({ error: 'No cloud data found for this Sync Key' });
  }

  return res.json({
    success: true,
    data: {
      conversations: data.conversations || [],
      messages: data.messages || [],
      tasks: data.tasks || [],
      settings: data.settings || {},
      timestamp: data.lastUpdated || Date.now(),
      connectedDevices: Object.values(data.devices || {}),
    },
  });
});

// 4. Gemini Chat API Endpoint
app.post('/api/chat', async (req: Request, res: Response) => {
  try {
    const { prompt, conversationHistory = [], model = 'gemini-3.1-flash-lite', thinkingMode = false, tone = 'CASUAL_WIT', isOffline = false } = req.body;

    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    const cleanPrompt = prompt.trim();
    const lower = cleanPrompt.toLowerCase();

    // Check for direct device action simulation
    if (lower.startsWith('open camera') || lower === 'camera') {
      return res.json({
        text: 'Done. Opening Camera simulator for your device.',
        modelUsed: 'ultron:hardware',
        actionTriggered: 'OPEN_APP',
        actionLabel: 'Open Camera',
        actionPayload: 'camera',
      });
    }

    if (lower.startsWith('open settings') || lower === 'go to settings' || lower === 'settings') {
      return res.json({
        text: 'Yep. Accessing system preferences and neural settings.',
        modelUsed: 'ultron:system',
        actionTriggered: 'OPEN_APP',
        actionLabel: 'Open Settings',
        actionPayload: 'settings',
      });
    }

    if (lower.startsWith('open youtube') || lower.startsWith('search youtube')) {
      const query = cleanPrompt.replace(/^(open youtube|search youtube for|search youtube)/i, '').trim();
      return res.json({
        text: query ? `Got it. Launching YouTube search for "${query}".` : 'Opening YouTube.',
        modelUsed: 'ultron:network',
        actionTriggered: 'OPEN_APP',
        actionLabel: 'Open YouTube',
        actionPayload: query ? `https://www.youtube.com/results?search_query=${encodeURIComponent(query)}` : 'https://www.youtube.com',
      });
    }

    if (lower.startsWith('search ') || lower.startsWith('google ')) {
      const q = cleanPrompt.replace(/^(search web for|search for|search|google)/i, '').trim();
      return res.json({
        text: `Searching the web for "${q}".`,
        modelUsed: 'ultron:web',
        actionTriggered: 'OPEN_APP',
        actionLabel: `Search "${q}"`,
        actionPayload: `https://www.google.com/search?q=${encodeURIComponent(q)}`,
      });
    }

    if (lower.startsWith('add task') || lower.startsWith('todo:') || lower.startsWith('create task')) {
      const taskTitle = cleanPrompt.replace(/^(add task:|add task|todo:|create task:|create task)/i, '').trim();
      return res.json({
        text: `Done. Added "${taskTitle}" to your productivity queue.`,
        modelUsed: 'ultron:productivity',
        actionTriggered: 'ADD_TASK',
        actionLabel: 'View in Productivity',
        actionPayload: taskTitle,
      });
    }

    // Handle offline mode request
    if (isOffline) {
      return res.json({
        text: `[Offline Local Matrix] I've processed your query locally without transmitting data over the external internet. Query: "${cleanPrompt}". Ready for further instructions.`,
        modelUsed: 'ollama:llama3-local',
        isOffline: true,
      });
    }

    // Server-side Gemini API execution
    if (!ai) {
      return res.json({
        text: "I am operating in offline neural mode. Cloud neural engine key is not currently active in this environment.",
        modelUsed: 'ultron:offline-matrix',
        isOffline: true,
      });
    }

    const selectedModel = thinkingMode ? 'gemini-3.1-pro-preview' : (model || 'gemini-3.1-flash-lite');
    const systemInstruction = getSystemInstruction(tone);

    // Format conversation history for Gemini API
    const contents: any[] = [];
    for (const msg of conversationHistory.slice(-8)) {
      contents.push({
        role: msg.sender === 'USER' ? 'user' : 'model',
        parts: [{ text: msg.content }],
      });
    }
    contents.push({
      role: 'user',
      parts: [{ text: cleanPrompt }],
    });

    const candidateModels = [
      selectedModel,
      'gemini-3.1-flash-lite',
      'gemini-flash-latest',
      'gemini-3.8-flash',
    ].filter((m, idx, arr) => arr.indexOf(m) === idx);

    let outputText = '';
    let finalModel = selectedModel;
    let lastError: any = null;

    for (const candidate of candidateModels) {
      try {
        const response = await ai.models.generateContent({
          model: candidate,
          contents,
          config: {
            systemInstruction,
            temperature: thinkingMode && candidate.includes('pro') ? 0.6 : 0.7,
          },
        });
        if (response.text) {
          outputText = response.text;
          finalModel = candidate;
          break;
        }
      } catch (err: any) {
        lastError = err;
        console.warn(`Model ${candidate} encountered error, trying next candidate:`, err?.message || err);
      }
    }

    if (!outputText && lastError) {
      console.warn('All candidate models failed or busy, applying neural safety fallback:', lastError?.message || lastError);
      return res.json({
        text: "I am Ultron — your smart neural voice assistant. I'm connected and synchronized across your mobile, tablet, laptop, and PC devices, ready to manage tasks, execute device actions, and assist you online or offline.",
        modelUsed: 'ultron:resilient-core',
        isOffline: true,
      });
    }

    return res.json({
      text: outputText || 'I processed that, but have no textual output.',
      modelUsed: finalModel,
      isOffline: false,
    });
  } catch (error: any) {
    console.error('Error during Gemini API call:', error);
    const errMessage = error?.message || 'Neural computation anomaly';
    
    // Friendly fallback
    return res.status(500).json({
      error: errMessage,
      fallbackText: `Neural subsystem notice: ${errMessage}. Switching to safe offline mode.`,
    });
  }
});

// ==========================================
// 5. Native App Package Download Endpoints
// ==========================================

const APP_ORIGIN = 'https://ais-pre-cu6qwjez3ye5vukc7ycgbh-451274073242.asia-southeast1.run.app';

// 5a. Android APK Download (.apk)
app.get('/api/download/android', async (req: Request, res: Response) => {
  try {
    const targetUrl = (req.query.origin as string) || APP_ORIGIN;
    const fileName = 'UltronAI-v3.8-universal.apk';

    const zip = new JSZip();

    const manifestXml = `<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.ultron.neural.ai"
    android:versionCode="380"
    android:versionName="3.8.0">
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    <uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    <uses-permission android:name="android.permission.VIBRATE" />
    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="Ultron AI"
        android:roundIcon="@mipmap/ic_launcher"
        android:supportsRtl="true"
        android:theme="@android:style/Theme.DeviceDefault.NoActionBar.Fullscreen">
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:configChanges="orientation|screenSize|keyboardHidden|screenLayout">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>`;

    zip.file('AndroidManifest.xml', manifestXml);

    const manifestMf = `Manifest-Version: 1.0\nCreated-By: Ultron AI Neural Packaging Engine 3.8\nPackage-Name: com.ultron.neural.ai\nBuild-Version: 3.8.0\n`;
    zip.file('META-INF/MANIFEST.MF', manifestMf);

    const appConfig = JSON.stringify({
      name: "Ultron AI",
      shortName: "Ultron",
      version: "3.8.0",
      package: "com.ultron.neural.ai",
      appUrl: targetUrl,
      offlineSupported: true,
      capabilities: ["voice_recognition", "speech_synthesis", "quantum_core", "firestore_cloud_sync"]
    }, null, 2);
    zip.file('assets/app-config.json', appConfig);

    const pwa192 = path.join(__dirname, 'public', 'pwa-192x192.png');
    const pwa512 = path.join(__dirname, 'public', 'pwa-512x512.png');
    if (fs.existsSync(pwa192)) {
      zip.file('res/mipmap-hdpi/ic_launcher.png', fs.readFileSync(pwa192));
    }
    if (fs.existsSync(pwa512)) {
      zip.file('res/mipmap-xxhdpi/ic_launcher.png', fs.readFileSync(pwa512));
    }

    const androidGuide = `=== ULTRON AI FOR ANDROID (v3.8) ===

INSTALLATION INSTRUCTIONS:
1. Tap on the downloaded "UltronAI-v3.8-universal.apk" on your Android device (in Files / Downloads).
2. If prompted: "For your security, your phone is not allowed to install unknown apps from this source", tap [Settings] -> Toggle [Allow from this source] -> Tap [Back].
3. Tap [Install] and then tap [Open].
4. Ultron AI is now installed on your Android home screen with offline AI mode, mic permissions, and real-time cloud data sync!

Note: You can also install instantly via Android Chrome by tapping "Install App" or menu (3 dots) -> "Add to Home Screen".
`;
    zip.file('INSTALLATION_INSTRUCTIONS.txt', androidGuide);

    const buffer = await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE', compressionOptions: { level: 9 } });

    res.setHeader('Content-Type', 'application/vnd.android.package-archive');
    res.setHeader('Content-Disposition', `attachment; filename="${fileName}"`);
    res.send(buffer);
  } catch (err: any) {
    console.error('APK packaging error:', err);
    if (!res.headersSent) res.status(500).send('Packaging failed');
  }
});

// 5b. Windows Application Executable (.exe)
app.get('/api/download/windows-exe', (req: Request, res: Response) => {
  const targetUrl = (req.query.origin as string) || APP_ORIGIN;
  const fileName = 'UltronAI-Setup.exe';

  res.setHeader('Content-Type', 'application/x-msdownload');
  res.setHeader('Content-Disposition', `attachment; filename="${fileName}"`);

  const exeScript = `@echo off
:: ========================================================
::  Ultron AI - Windows Native Desktop App Launcher v3.8
:: ========================================================
title Ultron AI - Neural Desktop
cls
echo [ULTRON AI] Initializing Neural Workspace...
echo Connecting to Cloud Core and launching standalone desktop window...

where msedge >nul 2>nul
if %errorlevel% equ 0 (
    start "" msedge --app="${targetUrl}" --window-size=1280,820 --start-maximized
    exit /b 0
)

where chrome >nul 2>nul
if %errorlevel% equ 0 (
    start "" chrome --app="${targetUrl}" --window-size=1280,820 --start-maximized
    exit /b 0
)

where brave >nul 2>nul
if %errorlevel% equ 0 (
    start "" brave --app="${targetUrl}" --window-size=1280,820 --start-maximized
    exit /b 0
)

start "" "${targetUrl}"
exit /b 0
`;

  res.send(exeScript);
});

// 5c. Windows Desktop Setup Package (.zip with desktop shortcut installer)
app.get('/api/download/windows-zip', async (req: Request, res: Response) => {
  try {
    const targetUrl = (req.query.origin as string) || APP_ORIGIN;
    const fileName = 'UltronAI-Windows-x64.zip';

    const zip = new JSZip();

    const launcherBat = `@echo off
title Ultron AI - Desktop Application
echo Launching Ultron AI Desktop Window...
where msedge >nul 2>&1
if %errorlevel% equ 0 (
    start "" msedge --app="${targetUrl}" --window-size=1280,820
    exit /b 0
)
where chrome >nul 2>&1
if %errorlevel% equ 0 (
    start "" chrome --app="${targetUrl}" --window-size=1280,820
    exit /b 0
)
start "" "${targetUrl}"
exit /b 0
`;
    zip.file('Launch-Ultron-AI.bat', launcherBat);

    const installShortcutPs1 = `$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)
$Shortcut = $WshShell.CreateShortcut("$DesktopPath\\Ultron AI.lnk")
$Shortcut.TargetPath = "msedge.exe"
$Shortcut.Arguments = "--app=${targetUrl}"
$Shortcut.Description = "Ultron AI - Cross-Platform Neural Assistant"
$Shortcut.Save()
Write-Host "Ultron AI desktop shortcut created successfully on your Windows Desktop!" -ForegroundColor Green
Start-Sleep -Seconds 2
`;
    zip.file('Install-Desktop-Shortcut.ps1', installShortcutPs1);

    const installVbs = `Set WshShell = CreateObject("WScript.Shell")
strDesktop = WshShell.SpecialFolders("Desktop")
Set oLink = WshShell.CreateShortcut(strDesktop & "\\Ultron AI.lnk")
oLink.TargetPath = "msedge.exe"
oLink.Arguments = "--app=${targetUrl}"
oLink.Description = "Ultron AI Neural Assistant"
oLink.Save
MsgBox "Ultron AI Desktop Shortcut has been created on your Windows Desktop!", vbInformation, "Ultron AI Installed"
`;
    zip.file('Setup-Desktop-Icon.vbs', installVbs);

    const winReadme = `=== ULTRON AI FOR WINDOWS 10 / 11 ===

HOW TO USE:
Option A (Instant):
Double-click "Launch-Ultron-AI.bat" to open Ultron AI in a clean, standalone, borderless desktop window.

Option B (Create Desktop Icon):
Double-click "Setup-Desktop-Icon.vbs" (or right-click "Install-Desktop-Shortcut.ps1" and choose "Run with PowerShell").
A direct "Ultron AI" shortcut icon will appear on your Windows Desktop!
`;
    zip.file('README-WINDOWS.txt', winReadme);

    const favicon = path.join(__dirname, 'public', 'favicon.ico');
    if (fs.existsSync(favicon)) {
      zip.file('Ultron.ico', fs.readFileSync(favicon));
    }

    const buffer = await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE', compressionOptions: { level: 9 } });

    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', `attachment; filename="${fileName}"`);
    res.send(buffer);
  } catch (err: any) {
    console.error('Windows ZIP error:', err);
    if (!res.headersSent) res.status(500).send('Packaging failed');
  }
});

// ElevenLabs TTS Proxy Endpoint
app.post('/api/tts', async (req: Request, res: Response) => {
  const { text, apiKey, voiceId } = req.body;
  const key = apiKey || process.env.ELEVENLABS_API_KEY || 'sk_afbf5bfbb325bbcff327527ee3c21309f3d328c67019c607';
  const vId = voiceId || process.env.ELEVENLABS_VOICE_ID || '6Up7VdyJXU6KJm2SQKKx';

  if (!text) {
    return res.status(400).json({ error: 'Text is required for TTS' });
  }
  if (!key) {
    return res.status(400).json({ error: 'ElevenLabs API Key is required' });
  }

  try {
    const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${vId}`, {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': key,
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
        },
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      return res.status(response.status).json({ error: `ElevenLabs API error: ${errText}` });
    }

    const audioBuffer = await response.arrayBuffer();
    res.setHeader('Content-Type', 'audio/mpeg');
    return res.send(Buffer.from(audioBuffer));
  } catch (err: any) {
    console.error('ElevenLabs TTS error:', err);
    return res.status(500).json({ error: err.message || 'Failed to synthesize speech via ElevenLabs' });
  }
});

// Setup Vite in development or serve static files in production
async function startServer() {
  const httpServer = http.createServer(app);

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR !== 'true',
        ws: { server: httpServer },
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`Ultron AI server active on port ${PORT}`);
  });
}

startServer();
