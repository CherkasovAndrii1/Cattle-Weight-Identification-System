// src/components/Footer/Footer.jsx
import React from 'react';

const Footer = () => {
    // Стили лучше вынести
    const footerStyle = { backgroundColor: '#1e1e1e', padding: '20px 16px', borderTop: '1px solid #333', boxShadow: '0 -4px 12px rgba(0,0,0,0.1)' };
    const containerStyle = { maxWidth: '1200px', margin: '0 auto', textAlign: 'center', fontSize: '14px' };
    const copyrightStyle = { margin: '0 0 8px', background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontWeight: '600' };
    const hintStyle = { margin: 0, color: '#777', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' };

    return (
        <footer style={footerStyle}>
            <div style={containerStyle}>
                <p style={copyrightStyle}>© 2025 Image Processing Application using ML</p>
                <p style={hintStyle}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#777" strokeWidth="2">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                    </svg>
                    Supported formats: JPEG, PNG, GIF, WEBP
                </p>
            </div>
        </footer>
    );
};

export default Footer;