import React from 'react';
import { Box, Button, Typography, Container, Grid, Paper, useTheme } from '@mui/material';
import { Link } from 'react-router-dom';

const MainPage: React.FC = () => {
  const theme = useTheme();

  return (
    <Container maxWidth="md" style={{ textAlign: 'center', marginTop: '5vh' }}>
      <Typography variant="h2" gutterBottom color="textPrimary">
        Welcome to Soulx's Homepage
      </Typography>
      <Typography variant="h5" paragraph color="textPrimary">
        This is the main entry point of the application. Explore the features below!
      </Typography>
      <Button 
        variant="contained" 
        color="primary" 
        component={Link} 
        to="/dashboard" 
        style={{ margin: '20px' }}
      >
        Go to Dashboard
      </Button>
      
      <Grid container spacing={4}>
        <Grid item xs={12} sm={6}>
          <Paper elevation={3} style={{ padding: '20px', backgroundColor: theme.palette.background.paper }}>
            <Typography variant="h5" color="textPrimary">Feature 1</Typography>
            <Typography variant="body1" color="textPrimary">
              Description of Feature 1 goes here. This feature is designed to enhance your experience.
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6}>
          <Paper elevation={3} style={{ padding: '20px', backgroundColor: theme.palette.background.paper }}>
            <Typography variant="h5" color="textPrimary">Feature 2</Typography>
            <Typography variant="body1" color="textPrimary">
              Description of Feature 2 goes here. This feature is user-friendly and efficient.
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default MainPage;
