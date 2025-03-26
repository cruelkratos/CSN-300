import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import {BotMessageSquare} from "lucide-react"
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Button, { ButtonProps } from '@mui/material/Button';
import { styled } from '@mui/material/styles';
import { purple } from '@mui/material/colors';
import { orange } from '@mui/material/colors';
import TextareaAutosize from '@mui/material/TextareaAutosize';
import AnimatedBackground from './AnimatedBackground'
import './App.css'
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});
function App() {
  // const [count, setCount] = useState(0)
  const [input,setInput] = useState("");
  const [model,setModel] = useState("llama");
  const [story,setStory] = useState("");
  const setGPT = () =>{
    setModel("gpt")
  }
  const setLlama = () =>{
    setModel("llama")
  }
  const ColorButton = styled(Button)<ButtonProps>(({ theme }) => ({
    color: theme.palette.getContrastText(purple[500]),
    backgroundColor: purple[500],
    '&:hover': {
      backgroundColor: purple[700],
    },
  }));
  const ColorButton2 = styled(Button)<ButtonProps>(({ theme }) => ({
    color: theme.palette.getContrastText(purple[500]),
    backgroundColor: orange[500],
    '&:hover': {
      backgroundColor: orange[700],
    },
  }));
  const supportedBooks = ['Harry Potter Saga' , 'A Song of Ice and Fire' , ]
  const handleChange = (e) =>{
    setInput(e.target.value);
  }
  const handleSubmit = async () =>{
    console.log(input);
    console.log(model);
    try{
      const data = {
        inp: input
      };
      const response = await fetch(`http://localhost:8000/${model}`,{
        method : 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body : JSON.stringify(data)
      });
      const reply = await response.json();
      console.log(reply);
      setStory(reply);
    }
    catch (e){
      console.error(e);
    }
  }
  const handleKeyDown = (e) =>{
    if (e.key == 'Enter'){
      handleSubmit();
    }
  } 
  return (
    <ThemeProvider theme={darkTheme}>
      <AnimatedBackground />
      <div className= "navbar">
        <BotMessageSquare />
        <div className='tit'>
        
        <h3>LLaMALore</h3>
        </div>
        
      </div>
      <div className='hero'>  
      
      <h1>Next-Gen Storytelling with Transformer-Driven Intelligence</h1> 
      <h3 className="low">Enter the realm of AI-powered storytelling, where literature meets technology. Select a book, provide a prompt, and let our model weave an original tale in the style of your favorite literary masterpiece. Our tool recreates the tone, style, and essence of iconic authors, bringing new stories to life with every input.</h3>
      
      </div>
      <div className = "inp">
      <TextField label="Input Your Story Prompt" color="secondary" value={input} onChange={handleChange} sx={{width:1100}} onKeyDown={handleKeyDown} />
      <div className='book-choice'>
      <Autocomplete
      disablePortal
      options={supportedBooks}
      sx={{ width: 150 }}
      renderInput={(params) => <TextField {...params} label="Book" />}
      />
      </div>
      </div>
      <div className='he'>
      <ColorButton variant="contained" onClick={setGPT}>GPT-1 (10 M)</ColorButton>
      <ColorButton2 variant="contained" onClick={setLlama}>Llama (7B)</ColorButton2>
      </div>
      <TextField
        label="Story"
        variant="outlined"
        value={story}
        multiline
        minRows={3} // Minimum number of rows
        maxRows={10} // Maximum number of rows
        readOnly
        fullWidth
        sx={{
          '& .MuiInputBase-input': {
            // Additional styling for the input field
            fontFamily: 'Arial, sans-serif',
            fontSize: '1rem',
            lineHeight: '1.5',
            padding: '8px',
          },
        }}
      />
    </ThemeProvider>
  )
}

export default App
