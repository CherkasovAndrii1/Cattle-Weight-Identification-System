// src/pages/LoginPage.jsx
import React from 'react';
import { Link } from 'react-router-dom';
// Импортируем компонент формы входа из папки components/LoginForm
import LoginForm from '../components/LoginForm/LoginForm'; // <--- ИСПРАВЛЕННЫЙ ИМПОРТ

function LoginPage() {
  const pageStyle = {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      paddingTop: '40px',
  };
  const linkStyle = {
      marginTop: '20px',
      color: '#aaa',
      textDecoration: 'none'
  };

  return (
    <div style={pageStyle}>
      {/* Рендерим импортированный компонент */}
      <LoginForm />
      <Link to="/register" style={linkStyle}
          onMouseOver={(e) => e.target.style.textDecoration = 'underline'}
          onMouseOut={(e) => e.target.style.textDecoration = 'none'}
      >
        Don't have an account? Register
      </Link>
    </div>
  );
}
export default LoginPage;