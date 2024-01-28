import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import {ComfyApp} from "./scripts/app.ts";

const app = ComfyApp.getInstance();
window.app = app;
window.graph = app.graph;

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
      <App app={app}/>
  </React.StrictMode>,
)
