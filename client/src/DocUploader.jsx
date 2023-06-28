import React from 'react';
import axios from 'axios';

const DocUploader = () => {
  const handleSubmit = (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    axios.post('http://127.0.0.1:5000/upload_doc', formData)
    .then((response) => {
        console.log(response.data);
    })
    .catch((error) => {
        console.error(error);
    });
  };

  return (
    <section className="absolute">
        <form id="upload-folder" encType="multipart/form-data" onSubmit={handleSubmit}>
            <label htmlFor="file-input" className="display: inline-block mr-2 w-[120px]">Select Folder:</label>
            <input type="file" webkitdirectory="true" id="file-input" name="file" directory="" multiple/>
            <button type="submit">Upload</button>
        </form>

        <form id="upload-file" encType="multipart/form-data" onSubmit={handleSubmit}>
            <label htmlFor="file-input" className="display: inline-block mr-2 w-[120px]">Select Files:</label>
            <input type="file" id="file-input" name="file" multiple/>
            <button type="submit">Upload</button>
        </form>
    </section>
  );
};

export default DocUploader;
