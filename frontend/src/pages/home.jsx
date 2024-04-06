import React from "react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./styles.css";
import a  from "../images/download (1).jpeg"

const Home = () => {
  const API_URL = "https://d9be-34-126-100-141.ngrok-free.app";
  const navigate = useNavigate();

  const [acknowledged, setAcknowledged] = useState(false);
  const [videoLink, setVideoLink] = useState("");

  const handleChange = (e) => {
    setVideoLink(e.target.value);
  };

  const handleSubmit = async (e) => {
    try {
      e.preventDefault();
      const response = await axios.post(
        `${API_URL}/post`,
        {
          videoLink: videoLink,
        },
        {
          headers: {
            "ngrok-skip-browser-warning": "true",
          },
        }
      );
      console.log(response);
      setTimeout(() => setAcknowledged(true), 10000);
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <>
      <div className="main">
   <div className="main-1">
<h1 className="h1_1">Shuttle Sensei</h1>
<h2 className="p">Practise Badminton with me</h2>
<div className="div">
<p className="p">Enter link here</p>
<input></input>
</div>
<div className="div">
<p className="p">Enter your mail</p>
<input></input>
</div>
<br></br>
<div>
<button className="button">Submit</button>
<button className="button">Show Results</button>
</div>
   </div>
   <div className="main-2">
   <img src={a} height={700} width={800} alt='player1' />
   </div>
      </div>

      <div>home</div>
      <form>
        <label>Video Link</label>
        <input
          type="text"
          value={videoLink}
          onChange={handleChange}
          placeholder="Enter the Google Drive link of the video"
        />
        <input type="submit" onClick={handleSubmit} />
      </form>

      <button onClick={() => navigate("/compare")} disabled={!acknowledged}>
        Show Results
      </button>
    </>
  );
};

export default Home;
