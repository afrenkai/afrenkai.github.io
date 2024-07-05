// src/components/Sidebar.tsx
import React from 'react';
import './Sidebar.css';
import { closeSidebar, navigateToSection } from '../Utils/navigation';
const Sidebar: React.FC = () => {
    return (
        <div className="sidebar" id="sidebar">
            <a href="javascript:void(0)" className="close-btn" onClick={() => closeSidebar()}>&times;</a>
            <a href="#about-me" onClick={() => navigateToSection('#about-me')}>About Me</a>
            <a href="#experience" onClick={() => navigateToSection('#experience')}>Experience</a>
            <a href="#classes-taken" onClick={() => navigateToSection('#classes-taken')}>Classes Taken</a>
            <a href="#Languages" onClick={() => navigateToSection('#Languages')}>Languages</a>
            <a href="#Libraries" onClick={() => navigateToSection('#Libraries')}>Libraries</a>
        </div>
    );
};

export default Sidebar;
