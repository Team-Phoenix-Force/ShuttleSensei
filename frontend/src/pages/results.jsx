import React, { useState } from "react";
import { Link } from "react-router-dom";
import Dropdown from "../components/dropdown";
import Player2image from "../images/player2.png";
import Player1image from "../images/player1.webp";
import CroppedPlayer2Image from "../images/cropped_player2.png";
import rallyTimeDistimg from "../images/rally_time_dist_img.jpg";
import shotDistPlayer1img from "../images/shots_dist_p1.png";
import shotDistPlayer2img from "../images/shots_dist_p2.png";
import posDistp1img from "../images/pos_dist_p1_img.png";
import combinedPosDistimg from "../images/combined_pos_dist_img.jpg";
import winErrorShots from "../images/win_error_shots.png";

import "../styles/results.css";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";

const Results = () => {
  const [page, setPage] = useState("Compare");
  const pages = ["Compare", "Player1", "Player2"];

  const [rally, setRally] = useState({
    player1: {
      smash: 10,
      drop: 20,
      clear: 30,
      drive: 40,
    },
    player2: {
      smash: 10,
      drop: 20,
      clear: 30,
      drive: 40,
    },
  });

  const [rallyArray, setRallyArray] = useState([
    {
      player1: {
        smash: 10,
        drop: 20,
        clear: 30,
        drive: 40,
      },
      player2: {
        smash: 10,
        drop: 20,
        clear: 30,
        drive: 40,
      },
    },
    {
      player1: {
        smash: 10,
        drop: 20,
        clear: 30,
        drive: 40,
      },
      player2: {
        smash: 10,
        drop: 20,
        clear: 30,
        drive: 40,
      },
    },
    {
      player1: {
        smash: 10,
        drop: 20,
        clear: 30,
        drive: 40,
      },
      player2: {
        smash: 10,
        drop: 20,
        clear: 30,
        drive: 40,
      },
    },
  ]);

  const [summary, setSummary] = useState('');

  const totalShots = rallyArray.reduce((acc, rally) => {
    return {
      player1: {
        smash: acc.player1.smash + rally.player1.smash,
        drop: acc.player1.drop + rally.player1.drop,
        clear: acc.player1.clear + rally.player1.clear,
        drive: acc.player1.drive + rally.player1.drive,
      },
      player2: {
        smash: acc.player2.smash + rally.player2.smash,
        drop: acc.player2.drop + rally.player2.drop,
        clear: acc.player2.clear + rally.player2.clear,
        drive: acc.player2.drive + rally.player2.drive,
      },
    };
  });

  const rallies = rallyArray.map((rally, index) => {
    return `Rally ${index + 1}`;
  });

  const togglePage = (newPage) => {
    console.log("new page : ", newPage);
    setPage(newPage);
  };

  const toggleRally = (newRally) => {
    console.log("new rally : ", newRally);
    const rallyIndex = parseInt(newRally.split(" ")[1]) - 1;
    setRally(rallyArray[rallyIndex]);
  };

  return (
    <div className="results-page">
      <div className="dropdown-1_1">
        <Dropdown list={pages} toggleFunction={togglePage} />
      </div>
      <div className="compare-section-one">
        <img src={CroppedPlayer2Image} alt="player2" />
        <div className="div1">
          <h1>Player1 </h1>
          <h2 className="h5_1"> VS </h2>
          <h1>Player2</h1>
          <h4 className="h4_1">Comparative analysis</h4>
        </div>
    
        <img src={Player1image} alt='player1' />
      </div>

      <div className="compare-section-two">
        <div className="compare-section-two-left">
          <div className="compare-section-2">
            <div className="rallies">
              <Dropdown list={rallies} toggleFunction={toggleRally} />
              <h2 className="analyze">Analyze by</h2>
            </div>
            <Tabs>
              <TabList className="tablist-2">
                <Tab color="red">
                  {page === "Compare" ? "Rally Time" : "Attacking Pattern"}
                </Tab>
                <Tab>
                  {page === "Compare"
                    ? "Combined Position Distribution"
                    : "Position Distribution"}
                </Tab>
                <Tab>
                  {page === "Compare"
                    ? "Combined Shot Distribution"
                    : "Shot Distribution"}
                </Tab>
              </TabList>

              <TabPanel>
                {/* {page === 'Compare' && (
                )} */}
                <div className="tablist">
                  <img
                    src={rallyTimeDistimg}
                    width={900}
                    height={500}
                    alt="rally-time-distribution"
                  />
                </div>
              </TabPanel>

              <TabPanel>
                <div className="tablist">
                  <img
                    src={combinedPosDistimg}
                    width={900}
                    height={500}
                    alt="combined-pos-dist"
                  />
                </div>
              </TabPanel>

              <TabPanel>
                <div className="tablist">
                  <img
                    src={shotDistPlayer1img}
                    width={900}
                    height={500}
                    alt="shot-dist-p1"
                  />
                </div>
              </TabPanel>
            </Tabs>
            {page === 'Compare' && (
              <p> {summary} </p>
            )}
          </div>
        </div>

        <div className="compare-section-two-right p1">
        <div>
        <div className="table0">
        <table className="table1">
              <th>Player1</th>
              <tr>
                <td>Smash</td>
                <td> {totalShots.player1.smash} </td>
              </tr>
              <tr>
                <td>Drop</td>
                <td> {totalShots.player1.drop}</td>
              </tr>
              <tr>
                <td>Clear</td>
                <td>{totalShots.player1.clear}</td>
              </tr>
              <tr>
                <td>Drive</td>
                <td>{totalShots.player1.drive}</td>
              </tr>
            </table>
            <table className="table1">
              <th>Player2</th>
              <tr>
                <td>Smash</td>
                <td> {totalShots.player2.smash} </td>
              </tr>
              <tr>
                <td>Drop</td>
                <td> {totalShots.player2.drop}</td>
              </tr>
              <tr>
                <td>Clear</td>
                <td>{totalShots.player2.clear}</td>
              </tr>
              <tr>
                <td>Drive</td>
                <td>{totalShots.player2.drive}</td>
              </tr>
            </table>
            </div>
        </div>
         <div className="imgdiv">
      <img
        src={winErrorShots}
        className="winErrorShots"
   
        alt="win-error-shots"
      />
      </div>
        </div>
      </div>
    </div>
  );
};

export default Results;
