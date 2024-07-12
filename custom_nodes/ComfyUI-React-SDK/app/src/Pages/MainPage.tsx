import React from 'react';
import { Typography, Container, Grid, Paper, useTheme } from '@mui/material';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const MainPage: React.FC = () => {
  const theme = useTheme();

  return (
    <Container 
      maxWidth="md" 
      style={{ 
        textAlign: 'center', 
        marginTop: '10vh', 
        backgroundColor: theme.palette.background.paper, // Changed to paper for better visibility
        padding: '2rem', 
        borderRadius: '16px', 
        boxShadow: theme.shadows[3],
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center' 
      }}
    >
      <motion.div 
        initial={{ opacity: 0, y: -20 }} 
        animate={{ opacity: 1, y: 0 }} 
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h2" gutterBottom color="textPrimary">
          Welcome to Soulx's AI
        </Typography>
        <Typography variant="h5" paragraph color="textPrimary">
          This is the main entry point of the application. Explore the features below!
        </Typography>
      </motion.div>

      <Grid container spacing={4}>
        <Grid item xs={12} sm={6}>
          <motion.div 
            whileHover={{ scale: 1.05 }} 
            transition={{ duration: 0.3 }}
          >
            <Paper 
              elevation={3} 
              style={{ 
                padding: '20px', 
                backgroundColor: theme.palette.primary.main, 
                color: 'white',
                borderRadius: '16px',
                textAlign: 'center',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100%',
                boxShadow: theme.shadows[4],
              }}
              component={Link} 
              to="/dashboard"
            >
              <Typography variant="h5">이미지 생성 AI</Typography>
              <Typography variant="body1">
                이 기능을 사용하여 이미지 생성을 시작하세요. 다양한 옵션을 설정할 수 있습니다.
              </Typography>
            </Paper>
          </motion.div>
        </Grid>
        <Grid item xs={12} sm={6}>
          <motion.div 
            whileHover={{ scale: 1.05 }} 
            transition={{ duration: 0.3 }}
          >
            <Paper 
              elevation={3} 
              style={{ 
                padding: '20px', 
                backgroundColor: theme.palette.secondary.main, 
                color: 'white',
                borderRadius: '16px',
                textAlign: 'center',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100%',
                boxShadow: theme.shadows[4],
              }}
            >
              <Typography variant="h5">누끼 따기 AI</Typography>
              <Typography variant="body1">
                이미지를 업로드하면 배경을 자동으로 제거하여 인물이나 사물만 남길 수 있습니다.
              </Typography>
            </Paper>
          </motion.div>
        </Grid>
      </Grid>
    </Container>
  );
};

export default MainPage;
