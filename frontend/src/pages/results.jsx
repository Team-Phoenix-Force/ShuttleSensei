import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import Dropdown from '../components/dropdown'
import Player2image from '../images/player2.png'
import Player1image from '../images/player1.webp'
import CroppedPlayer2Image from '../images/cropped_player2.png'
import '../styles/results.css'
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';

const Results = () => {
  const [page, setPage] = useState('Compare');
  const [rally, setRally] = useState('1');

  const pages = ['Compare', 'Player1', 'Player2']
  const rallies = ['1', '2', '3', '4', '5']

  const [playerDataArray, setPlayerDataArray] = useState([
    { 
      id: 1,
      player: 'Player1',
      smash: 10,
      drop: 20,
      clear: 30,
      drive: 40,
    },
    {
      id: 2,
      player: 'Player2',
      smash: 10,
      drop: 20,
      clear: 30,
      drive: 40,
    },
  ]);

  const togglePage = (newPage) => {
    console.log('new page : ', newPage)
    setPage(newPage)
  }

  const toggleRally = (newRally) => {
    console.log('new rally : ', newRally)
    setRally(newRally)
  }

  return (
    <>
      <div className='compare-section1'>
        <div>
          <Dropdown list={pages} toggleFunction={togglePage} />
        </div>
        <img src={CroppedPlayer2Image} alt='player2' />

        <div>
          <h1>Player1 </h1>
          <h5> VS </h5>
          <h1>Player2</h1>
          <h5>Comparative analysis</h5>
        </div>
    
        <img src={Player1image} alt='player1' />
      </div>
      <h1 style={{color:'green'}}>{page}</h1>
      
      <Dropdown list={rallies} toggleFunction={toggleRally} />
      <h1 style={{color:'green'}}>{rally}</h1>
      <div className='compare-section-2'>
        <h3>Analyze by</h3>
        <Tabs>
          <TabList>
            <Tab>Attacking  pattern</Tab>
            <Tab>Shot type</Tab>
            <Tab>Winner and error shots</Tab>
          </TabList>

          <TabPanel>
            <img src={Player1image} alt='player1' />
          </TabPanel>

          <TabPanel>
            <table>
              <tr>
                <td>Smash</td>
                <td>10</td>
              </tr>
              <tr>
                <td>Drop</td>
                <td>20</td>
              </tr>
              <tr>
                <td>Clear</td>
                <td>30</td>
              </tr>
              <tr>
                <td>Drive</td>
                <td>40</td>
              </tr>
            </table>
          </TabPanel>

          <TabPanel>
            <img src={CroppedPlayer2Image} alt='player2' />
          </TabPanel>
        </Tabs>
      </div>

    </>
  )
}

export default Results