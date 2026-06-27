import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './assets/css/global.css'
import App from './App.jsx'
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap-icons/font/bootstrap-icons.css';
import "remixicon/fonts/remixicon.css";


createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
