import React, { useEffect } from 'react';
import { IconButton } from '@mui/material';
import { Brightness4 as MoonIcon, Brightness7 as SunIcon } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { useMediaQuery } from '@mui/material';

interface ToggleColorModeProps {
  colorMode: 'light' | 'dark';
  setColorMode: React.Dispatch<React.SetStateAction<'light' | 'dark'>>;
}

const ToggleColorMode: React.FC<ToggleColorModeProps> = ({ colorMode, setColorMode }) => {
  const theme = useTheme();
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');

  useEffect(() => {
    setColorMode(prefersDarkMode ? 'dark' : 'light');
  }, [prefersDarkMode, setColorMode]);

  const toggleColorMode = () => {
    setColorMode(prevMode => (prevMode === 'light' ? 'dark' : 'light'));
  };

  useEffect(() => {
    document.body.style.backgroundColor = colorMode === 'dark' 
      ? theme.palette.background.default 
      : '#fff';
  }, [colorMode, theme.palette.background.default]);

  return (
    <IconButton
      onClick={toggleColorMode}
      style={{
        position: 'absolute',
        top: '0',
        right: '0',
        margin: '1rem',
        color:colorMode === 'dark' ? 'white' : 'inherit',
      }}

    >
      {colorMode === 'dark' ? <SunIcon /> : <MoonIcon />}
    </IconButton>
  );
};

export default ToggleColorMode;
