// src/pages/NewsPage.jsx
import React from 'react';

function NewsPage() {
  // Стили для примера
  const pageStyle = { padding: '20px' };
  const newsItemStyle = {
      marginBottom: '25px',
      paddingBottom: '15px',
      borderBottom: '1px solid #333'
  };
  const titleStyle = { marginBottom: '10px', color: '#e0e0e0' };
  const dateStyle = { fontSize: '0.85em', color: '#888', marginBottom: '10px'};
  const contentStyle = { color: '#ccc', lineHeight: '1.6'};

  // --- Замените это на загрузку данных с API или из файла, если нужно ---
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
  // --- Конец блока данных для примера ---


  return (
    <div style={pageStyle}>
      <h1>Project news</h1>
      {newsItems.length > 0 ? (
        newsItems.map(item => (
          <article key={item.id} style={newsItemStyle}>
            <h2 style={titleStyle}>{item.title}</h2>
            <div style={dateStyle}>{item.date}</div>
            <p style={contentStyle}>{item.content}</p>
          </article>
        ))
      ) : (
        <p>No news yet.</p>
      )}
      {/*
        Если новости будут загружаться:
        if (isLoading) return <p>Загрузка новостей...</p>;
        if (error) return <p style={{color: 'red'}}>Ошибка загрузки новостей: {error}</p>;
      */}
    </div>
  );
}

export default NewsPage;