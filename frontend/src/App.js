import logo from './logo.svg';
import './App.css';
import { Route, Routes } from 'react-router-dom';
import Home from './pages/home';
import Player from './components/player';
import Results from './pages/results';

function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/player" element={<Player />} />
        <Route path="/results" element={<Results />} />
      </Routes>
    </>
  );
}

export default App;
