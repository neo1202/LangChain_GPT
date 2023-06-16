import React from 'react';
import axios from 'axios';

const DocUploader = () => {
  const handleSubmit = (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    axios.post('http://127.0.0.1:5000/upload_folder', formData)
    .then((response) => {
        console.log(response.data);
    })
    .catch((error) => {
        console.error(error);
        // Handle the error here
    });
  };

  return (
    <section className="absolute">
        <form id="upload-form" encType="multipart/form-data" onSubmit={handleSubmit}>
            <input type="file" webkitdirectory="true" id="file-input" name="file" directory="" multiple/>
            <button type="submit">Upload</button>
        </form>

        <form id="upload-form" encType="multipart/form-data" onSubmit={handleSubmit}>
            <input type="file" id="file-input" name="file" multiple/>
            <button type="submit">Upload</button>
        </form>
    </section>
  );
};

export default DocUploader;
