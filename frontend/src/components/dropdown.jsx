import React from 'react'

const Dropdown = ({ list, toggleFunction }) => {
  const handleChange = (event) => {
    const selectedValue = event.target.value;
    console.log(selectedValue); // This will log the selected value
    toggleFunction(selectedValue);
  }

  return (
    <select onChange={handleChange} className='dropdown-1' style={{borderRadius:"10px",padding:"10px 0px",color:"greenyellow",border:"2px solid black",marginTop:"15px",marginLeft:"80px",fontWeight:"bolder",backgroundColor:"black", height:"40px", width:"100px"}}>
    
      {list.map((item, index) => {
        return (
          <option 
            key={index} 
            value={item}
          >{item}</option>
        )
      })}
    </select>
  )
}

export default Dropdown