// src/components/NewsPage/NewsPage.jsx
import React from 'react';

// Примерные новости
const newsItems = [
    {
        id: 3,
        date: '2025-04-25',
        title: 'Beta Version Launch!',
        icon: '🚀', // Example icon
        content: 'We are excited to announce the launch of the beta version of our image processing service! Features like segmentation and key point extraction are available. Weight estimation will be available soon!'
    },
    {
        id: 2,
        date: '2025-04-20',
        title: 'Segmentation Model Improved',
        icon: '🧠',
        content: 'The segmentation model (v1.1) has been updated. The accuracy of object boundary detection on complex backgrounds has been improved.'
    },
    {
        id: 1,
        date: '2025-04-15',
        title: 'New Website Design',
        icon: '🎨',
        content: 'We are pleased to introduce the updated design of our application! We’ve worked hard to make it more user-friendly and visually appealing in dark mode.'
    }
];
const NewsPage = () => {
    // --- Стили (лучше вынести в CSS) ---
    const containerStyle = {
        backgroundColor: '#1e1e1e',
        borderRadius: '16px',
        boxShadow: '0 8px 20px rgba(0,0,0,0.2)',
        padding: '24px',
        border: '1px solid #333',
        animation: 'fadeIn 0.5s ease-out', // Используем существующую анимацию
        color: '#e0e0e0'
    };

    const titleStyle = {
        fontSize: '22px',
        fontWeight: '600',
        marginBottom: '24px', // Увеличим отступ
        color: '#e0e0e0',
        position: 'relative',
        display: 'inline-block',
        borderBottom: '2px solid transparent', // Убираем подчеркивание по умолчанию
         paddingBottom: '5px', // Небольшой отступ для градиентной линии
         background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)', // Градиент для текста
         WebkitBackgroundClip: 'text',
         WebkitTextFillColor: 'transparent',
    };

    const newsListStyle = {
        display: 'flex',
        flexDirection: 'column',
        gap: '20px' // Пространство между новостями
    };

    const newsItemStyle = {
        backgroundColor: '#252525',
        padding: '20px',
        borderRadius: '12px',
        border: '1px solid #333',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
        transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
        cursor: 'default' // Или 'pointer' если новость кликабельна
    };

     // Стиль для hover-эффекта (лучше через CSS :hover)
     const handleMouseOver = (e) => {
        e.currentTarget.style.transform = 'translateY(-3px)';
        e.currentTarget.style.boxShadow = '0 6px 16px rgba(0, 0, 0, 0.2)';
     };
     const handleMouseOut = (e) => {
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.1)';
     };

    const newsTitleStyle = {
        fontSize: '18px',
        fontWeight: '600',
        margin: '0 0 8px 0',
        color: '#f0f0f0', // Чуть светлее основного
        display: 'flex',
        alignItems: 'center',
        gap: '10px' // Отступ между иконкой и текстом
    };

    const newsMetaStyle = {
        fontSize: '13px',
        color: '#888', // Тусклый цвет для даты
        marginBottom: '12px',
        display: 'block' // Чтобы занимал всю строку
    };

    const newsContentStyle = {
        fontSize: '15px',
        lineHeight: '1.6',
        color: '#c0c0c0' // Чуть тусклее заголовка
    };

    return (
        <div style={containerStyle}>
            <h2 style={titleStyle}>Project news</h2>
            <div style={newsListStyle}>
                {newsItems.map(item => (
                    <div
                       key={item.id}
                       style={newsItemStyle}
                       onMouseOver={handleMouseOver}
                       onMouseOut={handleMouseOut}
                    >
                        <h3 style={newsTitleStyle}>
                            <span style={{fontSize: '20px'}}>{item.icon}</span>
                            {item.title}
                        </h3>
                        <span style={newsMetaStyle}>{item.date}</span>
                        <p style={newsContentStyle}>
                            {item.content}
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
};

// Не забудь создать src/components/NewsPage/index.js:
// export { default } from './NewsPage';
export default NewsPage;