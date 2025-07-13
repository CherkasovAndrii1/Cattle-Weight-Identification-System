// src/components/NotificationSystem/NotificationSystem.jsx
import React from 'react';

// Принимает текущее уведомление как проп
const NotificationSystem = ({ notification }) => {
    if (!notification) {
        return null;
    }

    // Стили лучше вынести
    const notificationStyle = {
        position: 'fixed', top: '20px', right: '20px',
        backgroundColor: '#3b82f6', color: 'white',
        padding: '12px 16px', borderRadius: '12px',
        boxShadow: '0 8px 16px rgba(0,0,0,0.3), 0 4px 6px rgba(59,130,246,0.3)',
        zIndex: 1000, display: 'flex', alignItems: 'center',
        animation: 'slideIn 0.3s ease, float 3s ease infinite alternate' // Анимации должны быть в CSS
    };
    const iconStyle = { marginRight: '8px' };

    return (
        <div style={notificationStyle}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" style={iconStyle}>
                <circle cx="10" cy="10" r="8" stroke="white" strokeWidth="2"/>
                <path d="M6 10L9 13L14 7" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            {notification}
        </div>
    );
};

export default NotificationSystem;