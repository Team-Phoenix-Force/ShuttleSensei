import React from 'react'
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Home = () => {
  const API_URL = "https://d9be-34-126-100-141.ngrok-free.app/post_example";
  const navigate = useNavigate();

  const [acknowledged, setAcknowledged] = useState(false);
  const [videoLink , setVideoLink] = useState("");

  const handleChange = (e) => {
    setVideoLink(e.target.value);
  };

  const handleSubmit = async (e) => {
    try {
      e.preventDefault();
      const response = await axios.post(API_URL, {
        videoLink: videoLink,
      },
      {
        headers: {
          "ngrok-skip-browser-warning": 'true'
      }});
      console.log(response);
      setTimeout(() => setAcknowledged(true), 10000);
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <>
    <div>home</div>
    <form> 
      <label>Video Link</label>
      <input
        type='text'
        value={videoLink}
        onChange={handleChange}
        placeholder='Enter the Google Drive link of the video'
      />
      <input
        type='submit'
        onClick={handleSubmit}
      />
    </form>

    <button 
      onClick={() => navigate("/compare")}
      disabled={!acknowledged}
    >
      Show Results
    </button>
  
    </>
  )
}

export default Home