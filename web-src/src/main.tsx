import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import {ComfyApp} from "./scripts/app.ts";

// We should probably have a ComfyAppContextProvider that wraps the entire app?
const app = ComfyApp.getInstance();
window.app = app;
window.graph = app.graph;

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
      <App app={app}/>
  </React.StrictMode>,
)
