// src/components/Header.tsx
import React from 'react';
import './Header.css';
import { openSidebar } from '../Utils/navigation';
const Header: React.FC = () => {
    return (
        <div className="header" id="header">
            <span className="open-btn" onClick={() => openSidebar()}>&#9776; Menu</span>
            <h1>Artem Frenk</h1>
            <p>Data Scientist | Machine Learning Engineer First, AI Engineer Second</p>
            <a href="https://github.com/afrenkai" className="link" target="_blank" rel="noopener noreferrer">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" className="icon" />
                My GitHub
            </a>
            <a href="https://www.linkedin.com/in/artem-frenk-281726236/" className="link" target="_blank" rel="noopener noreferrer">
                <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" className="icon" />
                My LinkedIn
            </a>
            <a href="mailto:afrenk.biz@gmail.com" className="link" target="_blank" rel="noopener noreferrer">
                <img src="https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_%282020%29.svg" alt="Gmail" className="icon" />
                Email Me
            </a>
        </div>
    );
};

export default Header;
