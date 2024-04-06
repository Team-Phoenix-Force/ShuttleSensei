import React, { useState, useEffect } from "react";
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
import combinedShotDistimg from "../images/combined_shot_dist_img.jpg";
import winErrorShots from "../images/win_error_shots.png";
import axios from "axios";

import "../styles/results.css";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";

const Results = () => {
  const API_URL = "https://d9be-34-126-100-141.ngrok-free.app";
  const [page, setPage] = useState("Compare");
  const pages = ["Compare", "Player1", "Player2"];

  const [results, setResults] = useState({
    positionDistribution : 
    "",
    positionDistributionP1: 
    "",
    rallyTimeDistribution : 
    "",
    shotTypeDistributionP1 : 
    "",
    shotsDistributionP1 : 
    "",
    shotsDistributionP2 : "",
    summary : 
    "**Player 1**\n\n**Strengths:**\n\n* Excellent shot anticipation\n* Strong clears\n* Good court coverage\n\n**Weaknesses:**\n\n* Footwork needs improvement\n* Struggles with high-speed rallies\n* Tends to make unforced errors\n\n**Improvement Plan:**\n\n* Focus on improving footwork drills to enhance speed and agility on the court.\n* Introduce interval training to enhance endurance and stamina for extended rallies.\n* Implement a strategy to minimize unforced errors by emphasizing accuracy and timing.\n\n**Match Analysis:**\n\n* **Rally 1:** Player 1 demonstrated good shot selection with a drop followed by clears. However, the drop was slightly short, allowing Player 2 to gain the advantage.\n* **Rally 6:** A well-executed drive led to Player 1's victory, showcasing their ability to control the pace of the game.\n* **Rally 13:** Player 1 executed a drop shot effectively, but their follow-up clear was too high, allowing Player 2 to regain control.\n\n**Player 2**\n\n**Strengths:**\n\n* Exceptional smashing power\n* Accurate drop shots\n* Quick and agile movement\n\n**Weaknesses:**\n\n* Limited backhand range\n* Tendency to overhit shots\n* Lacks consistency in clears\n\n**Improvement Plan:**\n\n* Introduce specific drills to enhance backhand range and control.\n* Implement a technique to improve shot consistency, emphasizing proper swing mechanics.\n* Emphasize the importance of pacing shots rather than relying solely on power.\n\n**Match Analysis:**\n\n* **Rally 7:** Player 2 demonstrated their exceptional drop shot ability, forcing Player 1 to commit errors.\n* **Rally 11:** A powerful smash and accurate clears showcased Player 2's well-rounded skills.\n* **Rally 15:** Player 2's relentless drop shots and clear shots proved to be too much for Player 1, ultimately leading to their victory.",
    rallyData : "",
    totalPoints: {
      1: 4,
      2: 11
    },
    totalShotTypesP1: {
      clear: 12,
      drive: 3,
      drop: 7,
      smash: 3
    },
    totalShotTypesP2: {
      clear: 12,
      drive: 0,
      drop: 8,
      smash: 0
    }
  });

  const [rally, setRally] = useState({});
  const [newRallyData, setNewRallyData] = useState([]);
  const [rallies, setRallies] = useState([]);
  const [rallyIndex, setRallyIndex] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${API_URL}/get`);
        console.log(response);
        const newData = {
          positionDistribution : URL.createObjectURL(new Blob([response.data.positionDistribution], { type: 'image/png' })),
          positionDistributionP1: URL.createObjectURL(new Blob([response.data.positionDistributionP1], { type: 'image/png' })),
          rallyTimeDistribution : URL.createObjectURL(new Blob([response.data.rallyTimeDistribution], { type: 'image/png' })),
          shotTypeDistributionP1 : URL.createObjectURL(new Blob([response.data.shotTypeDistributionP1], { type: 'image/png' })),
          shotsDistributionP1 : URL.createObjectURL(new Blob([response.data.shotsDistributionP1], { type: 'image/png' })),
          shotsDistributionP2 : URL.createObjectURL(new Blob([response.data.shotsDistributionP2], { type: 'image/png' })),
          summary : response.data.summary,
          rallyData : response.data.rallyData,
          totalPoints: response.data.totalPoints,
          totalShotTypesP1: response.data.totalShotTypesP1,
          totalShotTypesP2: response.data.totalShotTypesP2
        }

        setResults(newData);

        newRallyData = response.data.rallyData.map((rally, index) => {
          const player1Shots = rally.player1.shots.split(",");
          const player2Shots = rally.player2.shots.split(",");
          const player1ShotsFreq = {
            clear: 0,
            drive: 0,
            drop: 0,
            smash: 0
          };
          const player2ShotsFreq = {
            clear: 0,
            drive: 0,
            drop: 0,
            smash: 0
          };
          player1Shots.forEach(shot => {
            player1ShotsFreq[shot] += 1;
          });
          player2Shots.forEach(shot => {
            player2ShotsFreq[shot] += 1;
          });

          return {
            length: rally.rallyLength,
            winner: rally.winner,
            player1 : {
              clear: player1ShotsFreq.clear,
              drive: player1ShotsFreq.drive,
              drop: player1ShotsFreq.drop,
              smash: player1ShotsFreq.smash
            },
            player2 : {
              clear: player2ShotsFreq.clear,
              drive: player2ShotsFreq.drive,
              drop: player2ShotsFreq.drop,
              smash: player2ShotsFreq.smash
            }
          }
        }); 

        setNewRallyData(newRallyData);

        setRally(newRallyData[0]);

        setRallies(newRallyData.map((rally, index) => `Rally ${index + 1}`));

      } catch (error) {
        console.log(error);
      }
    };
    fetchData();
  }, []);

  const togglePage = (newPage) => {
    console.log("new page : ", newPage);
    setPage(newPage);
  };

  const toggleRally = (newRally) => {
    console.log("new rally : ", newRally);
    const rallyIndex = parseInt(newRally.split(" ")[1]) - 1;
    setRallyIndex(rallyIndex);
    setRally(results.rallyData[rallyIndex]);
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
              {page !== "Compare" && (
                <Dropdown list={rallies} toggleFunction={toggleRally} />
              )}
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
                <div className="tablist">
                  {page === 'Compare' ? (
                    <img
                      src={results.rallyTimeDistribution}
                      width={900}
                      height={500}
                      alt="rally-time-dist"
                    />
                  ) : (
                    <img
                      src={rallyTimeDistimg}
                      width={900}
                      height={500}
                      alt="rally-time-dist"
                    />
                  )}
                </div>
              </TabPanel>

              <TabPanel>
                <div className="tablist">
                  {page === 'Compare' ? (
                    <img
                      src={results.positionDistribution}
                      width={900}
                      height={500}
                      alt="combined-pos-dist"
                    />
                  ) : (
                    <img
                      src={combinedShotDistimg}
                      width={900}
                      height={500}
                      alt="combined-shot-dist"
                    />
                  )}
                </div>
              </TabPanel>

              <TabPanel>
                {page === 'Compare' ? (
                  <img
                    src={results.shotTypeDistribution}
                    width={900}
                    height={500}
                    alt="combined-pos-dist"
                  />
                ) : (
                  <table className="table1">
                    <th>Player2</th>
                    <tr>
                      <td>Clear</td>
                      <td> {page === 'Player1' ? newRallyData[rallyIndex].player1.clear : newRallyData[rallyIndex].player2.clear} </td>
                    </tr>
                    <tr>
                      <td>Drop</td>
                      <td> {page === 'Player1' ? newRallyData[rallyIndex].player1.drop : newRallyData[rallyIndex].player2.drop} </td>
                    </tr>
                    <tr>
                      <td>Drive</td>
                      <td> {page === 'Player1' ? newRallyData[rallyIndex].player1.drive : newRallyData[rallyIndex].player2.drive} </td>
                    </tr>
                    <tr>
                      <td>Smash</td>
                      <td> {page === 'Player1' ? newRallyData[rallyIndex].player1.smash : newRallyData.player2.smash} </td>
                    </tr>
                  </table>
                )}
              </TabPanel>
            </Tabs>
            {page === 'Compare' && (
              <p> {results.summary} </p>
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
              <td> {results.totalShotTypesP1.smash} </td>
            </tr>
            <tr>
              <td>Drop</td>
              <td> {results.totalShotTypesP1.drop}</td>
            </tr>
            <tr>
              <td>Clear</td>
              <td>{results.totalShotTypesP1.clear}</td>
            </tr>
            <tr>
              <td>Drive</td>
              <td>{results.totalShotTypesP1.drive}</td>
            </tr>
          </table>
          <table className="table1">
            <th>Player2</th>
            <tr>
              <td>Smash</td>
              <td> {results.totalShotTypesP2.smash} </td>
            </tr>
            <tr>
              <td>Drop</td>
              <td> {results.totalShotTypesP2.drop}</td>
            </tr>
            <tr>
              <td>Clear</td>
              <td>{results.totalShotTypesP2.clear}</td>
            </tr>
            <tr>
              <td>Drive</td>
              <td>{results.totalShotTypesP2.drive}</td>
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
