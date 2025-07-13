// src/pages/RegisterPage.jsx
import React from 'react';
import { Link } from 'react-router-dom';
// Импортируем компонент формы регистрации из папки components/RegisterForm
import RegisterForm from '../components/RegisterForm/RegisterForm.jsx'; // <--- ИСПРАВЛЕННЫЙ ИМПОРТ

function RegisterPage() {
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
      <RegisterForm />
       <Link to="/login" style={linkStyle}
           onMouseOver={(e) => e.target.style.textDecoration = 'underline'}
           onMouseOut={(e) => e.target.style.textDecoration = 'none'}
       >
        Already have an account? Log in
      </Link>
    </div>
  );
}
export default RegisterPage;