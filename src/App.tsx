import React from 'react';
import Header from './Components/Header';
import Sidebar from './Components/Sidebar';
import CustomCarousel from './Components/Carousel';
import './App.css';

const App: React.FC = () => {
  const languageImages = [
    { src: 'https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg', alt: 'Python' },
    { src: 'https://upload.wikimedia.org/wikipedia/commons/1/1b/R_logo.svg', alt: 'R' },
    { src: 'https://upload.wikimedia.org/wikipedia/commons/4/4c/Typescript_logo_2020.svg', alt: 'TypeScript' },
    { src: 'https://upload.wikimedia.org/wikipedia/commons/1/18/ISO_C%2B%2B_Logo.svg', alt: 'C++' },
    { src: 'https://upload.wikimedia.org/wikipedia/commons/6/6a/JavaScript-logo.png', alt: 'JavaScript' },
    { src: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQEc9A_S6BPxCDRp5WjMFEfXrpCu1ya2OO-Lw&s', alt: 'HTML' },
  ];

  const frameworkImages = [
    { src: 'https://upload.wikimedia.org/wikipedia/commons/2/20/Tensorflow-svgrepo-com.svg', alt: 'TensorFlow' },
    { src: 'https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg', alt: 'Keras' },
    { src: 'https://matplotlib.org/3.3.1/_static/logo2_compressed.svg', alt: 'matplotlib' },
    { src: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQcR5U16C8yXgBpl7-Bc7Itjx3_LRl425zINA&s', alt: 'React' },
    { src: 'https://upload.wikimedia.org/wikipedia/commons/2/29/Postgresql_elephant.svg', alt: 'PostgreSQL' },
    { src: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRPSLjxQyIGfldJvcw8Ac5FdbxlhJEv4-gPRQ&s', alt: 'Firebase' },
    { src: 'https://cxl.com/wp-content/uploads/2019/10/google-bigquery-logo-1.png', alt: 'BigQuery' },
  ];

  return (
      <div className="App">
        <Header />
        <Sidebar />
        <div className="content">
          <div className="section" id="about-me">
            <h2>About Me</h2>
            <p>Hello, my name is Artem Frenk. I am a Rising Junior at Worcester Polytechnic Institute in Worcester, MA. I major in Data Science, with a focus on Machine Learning and NLP. Below, you can find some projects I have done, technologies (packages, frameworks, languages, etc.) that I am proficient in, and some of my experience.</p>
          </div>
          <div className="section" id="experience">
            <h2>Experience</h2>
            <p>Lorem ipsum dolor sit amet...</p>
          </div>
          <div className="section" id="classes-taken">
            <h2>Classes Taken</h2>
            <p>Sed ut perspiciatis unde omnis iste natus...</p>
          </div>
          <div className="section" id="Languages">
            <h3>Languages</h3>
            <CustomCarousel images={languageImages} />
          </div>
          <div className="section" id="Libraries">
            <h3>Libraries/Frameworks</h3>
            <CustomCarousel images={frameworkImages} />
          </div>
        </div>
        <div className="footer">
          <p>&copy; Artem Frenk 2024</p>
        </div>
      </div>
  );
};

export default App;
