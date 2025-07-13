// src/context/ThemeContext.js
import React, { createContext, useState, useMemo, useContext } from 'react';

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
    // 'light' или 'dark'
    const [theme, setTheme] = useState(() => {
        // Пытаемся получить тему из localStorage или системных настроек
        const storedTheme = localStorage.getItem('app-theme');
        // const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches; // Для системной темы
        return storedTheme || 'dark'; // По умолчанию темная
    });

    const toggleTheme = () => {
        setTheme(prevTheme => {
            const newTheme = prevTheme === 'light' ? 'dark' : 'light';
            localStorage.setItem('app-theme', newTheme); // Сохраняем выбор
            return newTheme;
        });
    };

    // Используем useMemo для предотвращения лишних ререндеров потребителей контекста
    const value = useMemo(() => ({ theme, toggleTheme }), [theme]);

    return (
        <ThemeContext.Provider value={value}>
            {children}
        </ThemeContext.Provider>
    );
}

// Хук для удобного использования контекста
export function useTheme() {
    const context = useContext(ThemeContext);
    if (context === undefined) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
}