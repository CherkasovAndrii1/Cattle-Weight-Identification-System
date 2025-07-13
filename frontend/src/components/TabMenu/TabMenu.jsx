// src/components/TabMenu/TabMenu.jsx (Пример обновления)
import React from 'react';
import { NavLink } from 'react-router-dom';
// import { useAuth } from '../../context/AuthContext'; // Если нужно скрывать/показывать вкладки

// Компонент больше не принимает activeProcessingTab, setActiveProcessingTab
function TabMenu() {
    // const { isAuthenticated } = useAuth();

    // Стили можно оставить прежними или упростить
    const menuStyle = { display: 'flex', gap: '10px', padding: '10px 0', borderBottom: '1px solid #333', marginBottom: '20px', flexWrap: 'wrap' };
    const itemStyle = { padding: '8px 16px', border: '1px solid transparent', borderRadius: '6px', cursor: 'pointer', textDecoration: 'none', color: '#ccc', backgroundColor: 'transparent', fontSize: '1em', transition: 'background-color 0.2s, color 0.2s, border-color 0.2s' };
    const activeStyle = { color: '#fff', backgroundColor: '#333', borderColor: '#555' };

    // Функция стилизации для NavLink (упрощенный вариант)
    const getNavLinkStyle = ({ isActive }) => ({
        ...itemStyle,
        ...(isActive ? activeStyle : {})
    });

    return (
        <nav style={menuStyle}>
            {/* Теперь все элементы - это ссылки NavLink */}
            <NavLink to="/segmentation" style={getNavLinkStyle}>Сегментация</NavLink>
            <NavLink to="/keypoints" style={getNavLinkStyle}>Ключевые Точки</NavLink>
            <NavLink to="/weight" style={getNavLinkStyle}>Оценка Веса</NavLink>

            {/* Ссылки на историю и новости */}
            {/* {isAuthenticated && ( // Показываем только для вошедших */}
                <NavLink to="/history" style={getNavLinkStyle}>История</NavLink>
            {/* )} */}
            <NavLink to="/news" style={getNavLinkStyle}>Новости</NavLink>
        </nav>
    );
}

export default TabMenu;