import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ThemeProvider } from '@mui/material/styles';
import { lightTheme, darkTheme } from './theme';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

const Index: React.FC = () => {
  const [colorMode, setColorMode] = useState<'light' | 'dark'>('light');

  useEffect(() => {
    const savedMode = localStorage.getItem('colorMode') as 'light' | 'dark';
    setColorMode(savedMode ? savedMode : 'light');
  }, []);

  useEffect(() => {
    localStorage.setItem('colorMode', colorMode);
  }, [colorMode]);

  const theme = colorMode === 'dark' ? darkTheme : lightTheme;

  return (
    <ThemeProvider theme={theme}>
      <App colorMode={colorMode} setColorMode={setColorMode} />
    </ThemeProvider>
  );
};

root.render(
  <React.StrictMode>
    <Index />
  </React.StrictMode>
);

// Performance measuring
reportWebVitals();
