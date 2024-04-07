import React from "react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./styles.css";
import a  from "../images/download (1).jpeg"

const Home = () => {
  const API_URL = "https://7d47-35-237-195-78.ngrok-free.app";
  const navigate = useNavigate();

  const [acknowledged, setAcknowledged] = useState(false);
  const [videoLink, setVideoLink] = useState("");
  const [email, setEmail] = useState("");

  const handleChange = (e) => {
    setVideoLink(e.target.value);
  };

  const handleSubmit = async (e) => {
    try {
      e.preventDefault();
      const response = await axios.post(
        `${API_URL}/images`,
        {
          videoLink: videoLink,
          email: email,
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
<form>
        <label>Video Link</label>
        <input
          type="text"
          value={videoLink}
          onChange={handleChange}
          placeholder="Enter the Google Drive link of the video"
        />
        <br></br>
        <label>Email</label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Enter your email"
        />
        <input type="submit" onClick={handleSubmit} />
      </form>

      <button onClick={() => navigate("/results")} disabled={!acknowledged}>
        Show Results
      </button>
</div>
   </div>
   <div className="main-2">
   <img src={a} height={700} width={800} alt='player1' />
   </div>
      </div>
      
    </>
  );
};

export default Home;
