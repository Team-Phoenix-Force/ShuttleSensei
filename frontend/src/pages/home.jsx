import React from 'react'
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Home = () => {
  const [acknowledged, setAcknowledged] = useState(false);

  const navigate = useNavigate();

  const API_URL = "https://n0nauf9pvy-496ff2e9c6d22116-5000-colab.googleusercontent.com/";
  const [videoLink , setVideoLink] = useState("");

  const handleChange = (e) => {
    setVideoLink(e.target.value);
  };

  const handleSubmit = async (e) => {
    try {
      e.preventDefault();
      const response = await axios.get(`${API_URL}`);
      console.log(response);
      setAcknowledged(true);
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
    >Show Results</button>
  
    </>
  )
}

export default Home