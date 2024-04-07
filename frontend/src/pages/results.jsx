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
import winErrorShots1 from "../images/WinErrShots1.png";
import winErrorShots2 from "../images/WinErrShots2.png";
import winErrorShots3 from "../images/WinErrShots3.png";
import axios from "axios";
import ReactMarkdown from 'react-markdown';

import "../styles/results.css";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";

const Results = () => {  
  const API_URL = "http://localhost:8000/api/hackbyte/";
  const [page, setPage] = useState("Compare");
  const pages = ["Compare", "Player1", "Player2"];

  const [results, setResults] = useState({
    positionDistributionP1: 
    "",
    rallyTimeDistribution : 
    "",
    shotTypeDistributionP1 : 
    "",
    shotsDistributionP1 : 
    "",
    shotsDistributionP2 : "",
    summary : "",
    rallyData : [],
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
  const [img,setImg] = useState("");

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${API_URL}`,{
          headers: {
            "ngrok-skip-browser-warning": "true",
          },
          withCredentials: true,
        });
        console.log(response);
        setImg(response.data);
        const newData = {
          positionDistributionP1: response.data.images.position_distribution_p1,
          rallyTimeDistribution : response.data.images.rally_time_distribution,
          shotTypeDistributionP1 : response.data.images.shot_type_distribution_p1,
          shotsDistributionP1 : response.data.images.shots_distribution_p1,
          shotsDistributionP2 : response.data.images.shots_distribution_p2,
          summary : response.data.long_string,
          rallyData : response.data.rally_data,
          totalPoints: response.data.total_points,
          totalShotTypesP1: response.data.total_shot_types_p1,
          totalShotTypesP2: response.data.total_shot_types_p2
        }

        setResults(newData);

        console.log(newData);

        const newRallyArrayData = response.data.rally_data.map((rally, index) => {
          const player1Shots = rally['Player_1_Shots'].split(",");
          const player2Shots = rally['Player_2_Shots'].split(",");
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
          })

          return {
            length: rally['Rally_Length'],
            winner: rally['Winner'],
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
            },
            rallyNumber: index
          }
        })

        setNewRallyData(newRallyArrayData);

        setRally(newRallyArrayData[0]);

        setRallies(newRallyArrayData.map((rally, index) => `Rally ${index + 1}`));

        console.log("NEW RALLY DATA : ");

        console.log(newRallyArrayData);

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
              <h2 className="analyze"> In depth analysis</h2>
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
                      src={results.rallyTimeDistribution}
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
                      src={results.positionDistributionP1}
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
                    src={results.shotsDistributionP1}
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
                      <td> {page === 'Player1' ? newRallyData[rallyIndex].player1.smash : newRallyData[rallyIndex].player2.smash} </td>
                    </tr>
                  </table>
                )}
              </TabPanel>
            </Tabs>
            {page === 'Compare' && (
              <div style={{color:'black'}}>
                {/* <ReactMarkdown>{mdText} </ReactMarkdown> */}
                <p>{results.summary}</p>
              </div>
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
          <h3 style={{color:'#000', fontWeight: 'bold'}}> Winning & Error Shots</h3>

          <img
            src={ page==='Compare' ? winErrorShots1 : (page==='Player1' ? winErrorShots2 : winErrorShots3)}
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
