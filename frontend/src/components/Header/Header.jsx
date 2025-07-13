import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext'; // ПРАВИЛЬНО

function Header() {
    const { isAuthenticated, user, logout } = useAuth();
    const navigate = useNavigate();



 
    const headerStyle = {
        padding: '10px 30px 15px 30px', 
        backgroundColor: '#1f1f1f',
        color: '#e0e0e0',
        boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
        borderBottom: '1px solid #333',
    };


    const topRowStyle = {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '15px', 
        minHeight: '40px',
    };

   
    const titleStyle = {
        margin: 0,
        fontSize: '1.8rem',
        fontWeight: '600', // Сделаем пожирнее
        backgroundImage: 'linear-gradient(90deg, #ff8a00, #e52e71)', // Оранжево-розовый
        backgroundClip: 'text',
        WebkitBackgroundClip: 'text',
        color: 'transparent',
        backgroundColor: 'transparent', // Убедимся, что фон H1 прозрачный
        userSelect: 'none', // Запретим выделение текста заголовка
    };

    // Контейнер для элементов аутентификации справа
    const authLinksStyle = {
        display: 'flex',
        alignItems: 'center',
        gap: '15px',
    };

    // Стиль для ссылок "Вход" / "Регистрация"
    const authLinkStyle = {
        color: '#bb86fc', // Фиолетовый акцент
        textDecoration: 'none',
        fontSize: '0.9em',
        fontWeight: '500',
        transition: 'color 0.2s ease',
        ':hover': { // Псевдокласс hover не работает в inline-стилях, используйте CSS или JS
            color: '#ffffff',
        }
    };
    // Стиль для кнопки "Выход"
    const logoutButtonStyle = {
        background: 'none',
        border: 'none',
        color: '#aaa',
        cursor: 'pointer',
        fontSize: '0.9em',
        fontWeight: '500',
        padding: 0,
        transition: 'color 0.2s ease',
        ':hover': {
             color: '#fff',
        }
    };
    // Стиль для отображения email пользователя
    const userInfoStyle = {
        fontSize: '0.9em',
        color: '#bbb',
        marginRight: '10px', // Отступ перед кнопкой Выход
    };

    // Стиль нижней строки (Кнопка Назад + Навигация)
    const bottomRowStyle = {
        display: 'flex',
        alignItems: 'center',
        gap: '20px',
    };

    // Стиль кнопки "Назад"
    const backButtonStyle = {
        background: '#333', // Немного фона
        border: '1px solid #555',
        color: '#ccc',
        padding: '6px 12px',
        borderRadius: '6px',
        cursor: 'pointer',
        fontSize: '1.1em', // Сделаем стрелку чуть больше
        lineHeight: '1', // Убрать лишнюю высоту строки
        transition: 'background-color 0.2s ease, border-color 0.2s ease',
        ':hover': {
             backgroundColor: '#444',
             borderColor: '#777',
        }
    };

    // Стиль контейнера навигации
    const navStyle = {
        display: 'flex',
        gap: '12px', // Расстояние между кнопками
        flexWrap: 'wrap',
        alignItems: 'center',
    };

    // --- Стили Навигационных Кнопок/Ссылок ---
    const navLinkBaseStyle = {
        padding: '8px 22px', // Вертикальный / Горизонтальный паддинг
        borderRadius: '999px', // Овальная форма
        cursor: 'pointer',
        textDecoration: 'none',
        fontSize: '0.9em', // Размер шрифта внутри кнопки
        fontWeight: '500',
        border: 'none',
        transition: 'all 0.2s ease-out',
        display: 'inline-block',
        boxShadow: '0 3px 6px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.05)', // Тень для объема + легкий внутренний блик сверху
        textAlign: 'center',
    };

    const navLinkStyle = { // Неактивная ссылка
        ...navLinkBaseStyle,
        color: '#d0d0d0',
        backgroundImage: 'linear-gradient(145deg, #424242, #303030)', // Темно-серый градиент
    };

    const activeNavLinkStyle = { // Активная ссылка (добавляется к базовому)
        color: '#ffffff',
        backgroundImage: 'linear-gradient(145deg, #8a4bf1, #c73879)', // Фиолетово-розовый градиент
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.5), inset 0 1px 2px rgba(0, 0, 0, 0.3)', // Тень "внутрь"
        transform: 'translateY(1px)', // Эффект нажатия
    };

    // Функция для получения стилей NavLink
    const getNavLinkStyle = ({ isActive }) => ({
        ...(isActive ? { ...navLinkStyle, ...activeNavLinkStyle } : navLinkStyle)
    });

    // --- Обработчики ---
    const handleGoBack = () => {
        navigate(-1); // Перейти назад по истории браузера
    };

    // --- Рендеринг Компонента ---
    return (
        <header style={headerStyle}>
            {/* Верхняя строка: Заголовок и Статус Входа */}
            <div style={topRowStyle}>
                <h1 style={titleStyle}>Image Processor</h1>
                <div style={authLinksStyle}>
                    {isAuthenticated ? (
                        <>
                            {user && user.email && <span style={userInfoStyle}>{user.email}</span>}
                            {/* Используем NavLink для единообразия стилей, но можно и button */}
                            <NavLink to="#" onClick={(e) => { e.preventDefault(); logout(); }} style={{...authLinkStyle, color: '#aaa'}}>Log out</NavLink>
                            {/* <button onClick={logout} style={logoutButtonStyle}>Выход</button> */}
                        </>
                    ) : (
                        <>
                            <NavLink to="/login" style={authLinkStyle}>Sign in</NavLink>
                            <NavLink to="/register" style={authLinkStyle}>Registration</NavLink>
                        </>
                    )}
                </div>
            </div>

            {/* Нижняя строка: Кнопка Назад и Навигация */}
            <div style={bottomRowStyle}>
                <button onClick={handleGoBack} style={backButtonStyle} title="Назад">
                    &larr; {/* HTML-символ стрелки влево */}
                </button>
                <nav style={navStyle}>
                    <NavLink to="/segmentation" style={getNavLinkStyle}>Segmentation</NavLink>
                    <NavLink to="/keypoints" style={getNavLinkStyle}>Extracting keypoints</NavLink>
                    <NavLink to="/weight" style={getNavLinkStyle}>Weight estimation</NavLink>
                    {}
                    {isAuthenticated && (
                        <NavLink to="/history" style={getNavLinkStyle}>Request History</NavLink>
                    )}
                    <NavLink to="/news" style={getNavLinkStyle}>News</NavLink>
                </nav>
            </div>
        </header>
    );
}

export default Header;