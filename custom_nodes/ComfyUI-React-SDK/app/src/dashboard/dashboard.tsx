import React, { useEffect, useState } from 'react';
import { useComfy } from '../comfy/ComfyProvider';
import {
  Box,
  Button,
  Select,
  MenuItem,
  Stack,
  TextField,
  LinearProgress,
  FormControl,
  FormLabel,
  FormControlLabel,
  Checkbox,
  Slider,
  useTheme,
} from '@mui/material';
import { Subscribe, UnSubscribe, WS_MESSAGE_TYPE_EXECUTED, WS_MESSAGE_TYPE_PROGRESS } from '../comfy/api';
import { base } from './image';
import { Image } from '@mui/icons-material';

interface DashboardProps {

}

const Dashboard: React.FC<DashboardProps> = () => {
  const theme = useTheme();
  const { queuePrompt, fetchCheckpoints } = useComfy();

  const [rand, setRand] = useState<number>(Math.random());
  const [image, setImage] = useState<string | null>(null);
  const [checkpoints, setCheckpoints] = useState<string[]>([]); // 수정된 부분
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('');
  
  const [cfg, setCfg] = useState(5);
  const [steps, setSteps] = useState(25);
  const [seed, setSeed] = useState(Math.round(Math.random() * Number.MAX_SAFE_INTEGER));
  const [randomSeed, setRandomSeed] = useState(true);
  const [height, setHeight] = useState(512);
  const [width, setWidth] = useState(512);
  const [positivePrompt, setPositivePrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');

  const [inProgress, setInProgress] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    updateCheckpoint();
    Subscribe('dashboard', (event) => {
      const message = JSON.parse(event.data);
      if (message.type === WS_MESSAGE_TYPE_EXECUTED) {
        setRand(Math.random());
        setImage(message.data.output.images[0].filename);
        setInProgress(false);
        setProgress(0);
      } else if (message.type === WS_MESSAGE_TYPE_PROGRESS) {
        setProgress(Math.floor((message.data.value / message.data.max) * 100));
      }
    });
    return () => {
      UnSubscribe('dashboard');
    };
  }, []);

  const updateCheckpoint = () => {
    fetchCheckpoints().then((fetchedCheckpoints) => {
      setCheckpoints(fetchedCheckpoints[0]); // 수정된 부분
    });
  };

  const generate = () => {
    if (!selectedCheckpoint) {
      alert('No Checkpoint is selected');
      return;
    }
    queuePrompt({
      cfg,
      steps,
      seed,
      checkpoint: selectedCheckpoint,
      height,
      width,
      positivePrompt,
      negativePrompt,
    }).then((res) => {
      if (res.prompt_id) {
        alert(`Prompt Submitted: ${res.prompt_id}`);
        setInProgress(true);
      }
    });

    if (randomSeed) {
      setSeed(Math.round(Math.random() * Number.MAX_SAFE_INTEGER));
    }
  };

  return (
    <div>
      <Stack direction="row" spacing={2} style={{ width: '100%' }}>

        <Box flex="1" style={{ padding: '20px' }}>
          <Stack direction="column" spacing={6} style={{ marginTop: '5vh' }}>

            <img src='./SoulxLogo.png'/>
            <FormControl fullWidth>
              <Select value={selectedCheckpoint} onChange={(e) => setSelectedCheckpoint(e.target.value)} displayEmpty>
                <MenuItem value="" disabled>Select Checkpoint</MenuItem>
                {checkpoints.map((option, index) => (
                  <MenuItem key={index} value={option}>{option}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl>
              <FormLabel>CFG ({cfg})</FormLabel>
              <Slider value={cfg} min={1} max={10} step={0.5} onChange={(e, value) => setCfg(value as number)} valueLabelDisplay="auto" />
            </FormControl>
            <FormControl>
              <FormLabel>Steps ({steps})</FormLabel>
              <Slider value={steps} min={1} max={100} step={1} onChange={(e, value) => setSteps(value as number)} valueLabelDisplay="auto" />
            </FormControl>
            <Stack direction="row" spacing={5}>
              <TextField
                label="Seed"
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                InputProps={{ inputProps: { min: 1, max: Number.MAX_SAFE_INTEGER } }}
                fullWidth
              />
              <FormControlLabel
                control={<Checkbox checked={randomSeed} onChange={(e) => setRandomSeed(e.target.checked)} />}
                label="Random Seed"
              />
            </Stack>
            <Stack direction="row" spacing={5}>
              <TextField label="Height" type="number" value={height} onChange={(e) => setHeight(Number(e.target.value))} fullWidth />
              <TextField label="Width" type="number" value={width} onChange={(e) => setWidth(Number(e.target.value))} fullWidth />
            </Stack>
            <TextField label="Positive Prompt" multiline rows={4} value={positivePrompt} onChange={(e) => setPositivePrompt(e.target.value)} fullWidth />
            <TextField label="Negative Prompt" multiline rows={4} value={negativePrompt} onChange={(e) => setNegativePrompt(e.target.value)} fullWidth />
            <Button variant="contained" color="primary" onClick={generate}>Generate</Button>
            {inProgress && <LinearProgress variant="determinate" value={progress} />}
          </Stack>
        </Box>
        <Box flex="2" display="flex" alignItems="center" justifyContent="center" height="100vh">
          <Box
            component="div"
            minWidth="80%"
            maxWidth="80%"
            border="2px solid black"
            p="4"
            borderRadius="md"
            style={{ aspectRatio: '1/1' }}
          >
            {image ? (
              <img
                src={`/view?filename=${image}&type=output&rand=${rand}`}
                alt=""
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              />
            ) : (
              <img src={base} alt="Red dot" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            )}
          </Box>
        </Box>
        <Box flex="1"></Box>
      </Stack>
    </div>
  );
};

export default Dashboard;
