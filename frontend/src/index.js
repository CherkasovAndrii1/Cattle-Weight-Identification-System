// src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx'; // Убедитесь, что импортируется App.jsx
import './styles/global.css';
import './styles/animations.css';
import './index.css';
import { AuthProvider } from './context/AuthContext'; // Импорт провайдера

const rootElement = document.getElementById('root');
const root = ReactDOM.createRoot(rootElement);



root.render(
  <React.StrictMode>
    <AuthProvider>   {/* <<<=== ПРОВЕРЬТЕ ЭТУ ОБЕРТКУ ===>>> */}
      <App />
    </AuthProvider>  {/* <<<=== И ЗАКРЫТИЕ ===>>> */}
  </React.StrictMode>
);