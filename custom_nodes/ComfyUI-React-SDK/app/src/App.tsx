import React from 'react';
import './App.css';
import Dashboard from './dashboard/dashboard';
import { ComfyProvider } from './comfy/ComfyProvider';

interface AppProps {
  colorMode: 'light' | 'dark';
  setColorMode: React.Dispatch<React.SetStateAction<'light' | 'dark'>>;
}

const App: React.FC<AppProps> = ({ colorMode, setColorMode }) => {
  return (
    <ComfyProvider>
      <Dashboard colorMode={colorMode} setColorMode={setColorMode} />
    </ComfyProvider>
  );
};

export default App;
