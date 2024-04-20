import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./styles.css";
import badmintonImage from "../images/badmintonMatchImg.jpeg";
import { POST_URL } from "../constants";

const Home = () => {
	const navigate = useNavigate();

	const [videoLink, setVideoLink] = useState("");
	const [email, setEmail] = useState("");

	const handleChange = (e) => {
		if (e.target.name === "videoLink") {
			setVideoLink(e.target.value);
		} else {
			setEmail(e.target.value);
		}
	};

	const handleSubmit = async (e) => {
		try {
			e.preventDefault();
			const response = await axios.post(
				POST_URL,
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
		} catch (error) {
			console.log(error);
		}
	};

	return (
		<div
			className="container"
			style={{
				background: "linear-gradient(120deg, #fdfbfb 0%, #74C6EF 100%)",
			}}
		>
			<div className="content">
				<h1 className="title">Shuttle Sensei</h1>
				<h2 className="subtitle">Practice Badminton with Me</h2>
				<form className="form" onSubmit={handleSubmit}>
					<div className="form-group">
						<input
							type="text"
							id="videoLink"
							name="videoLink"
							value={videoLink}
							onChange={handleChange}
							placeholder="Enter the Google Drive link of the video"
							className="input-field"
						/>

						<input
							type="email"
							id="email"
							name="email"
							value={email}
							onChange={handleChange}
							placeholder="Enter your email address"
							className="input-field"
						/>
					</div>
					<button
						className="button submit-button"
						type="submit"
						disabled={!videoLink}
					>
						Submit
					</button>
					<button
						className="button show-results"
						onClick={() => navigate("/results")}
						disabled={!videoLink}
					>
						Show Results
					</button>
				</form>
			</div>
			<div className="image-container">
				<img src={badmintonImage} alt="Badminton Player" />
			</div>
		</div>
	);
};

export default Home;
