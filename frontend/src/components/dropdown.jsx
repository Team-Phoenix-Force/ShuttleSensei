import React from 'react'

const Dropdown = ({ list, toggleFunction }) => {
  const handleChange = (event) => {
    const selectedValue = event.target.value;
    console.log(selectedValue); // This will log the selected value
    toggleFunction(selectedValue);
  }

  return (
    <select onChange={handleChange}>
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