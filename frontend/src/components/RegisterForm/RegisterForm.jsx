// src/components/RegisterForm.jsx (Пример с MUI)
import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext'; // Убедитесь, что путь верный
import { useNavigate, Link as RouterLink } from 'react-router-dom';

// MUI Компоненты
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import Alert from '@mui/material/Alert';
import Link from '@mui/material/Link';
import CircularProgress from '@mui/material/CircularProgress';
import { useNotification } from '../NotificationSystem/NotificationSystem'; // Переконайтесь, що шлях правильний

// (Опционально) Иконки
// import PersonAddAltIcon from '@mui/icons-material/PersonAddAlt';
// import Avatar from '@mui/material/Avatar';

function RegisterForm() {
    const { register, isLoading, authError } = useAuth();
    const navigate = useNavigate();

    // Локальное состояние
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [localError, setLocalError] = useState('');

    // Обработчик отправки
    const handleSubmit = async (event) => {
        event.preventDefault();
        setLocalError(''); // Сброс локальной ошибки

        // Валидация на клиенте
        if (!email || !password || !confirmPassword) {
            setLocalError('Please fill all fields');
            return;
        }
        if (!/\S+@\S+\.\S+/.test(email)) {
            setLocalError('Please enter correct email');
            return;
        }
        if (password !== confirmPassword) {
            setLocalError('Password mismatch');
            return;
        }
        if (password.length < 6) {
            setLocalError('Password length should be greater than 6');
            return;
        }

        // Вызов функции регистрации из контекста
        const success = await register({ email, password });

        if (success) {
            console.log("Registration successful, navigating to login...");
            // Показываем уведомление (можно заменить на вашу систему уведомлений useNotification)
            navigate('/login'); // Перенаправляем на страницу входа
        }
        // Если !success, ошибка будет в authError и отобразится ниже
    };

    // Определяем общую ошибку для подсветки полей и Alert
    const isError = Boolean(localError || authError);
    const errorMessage = localError || authError || '';

    return (
        <Container component="main" maxWidth="xs"> {/* Центрирование и ограничение ширины */}
            <Box
                sx={{
                    marginTop: 8,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    bgcolor: 'background.paper',
                    padding: 4,
                    borderRadius: 2,
                    boxShadow: 3,
                }}
            >
                {/* (Опционально) Иконка/Аватар */}
                {/* <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
                    <PersonAddAltIcon />
                </Avatar> */}
                <Typography component="h1" variant="h5">
                    Registration
                </Typography>
                <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 3, width: '100%' }}> {/* Увеличили отступ сверху формы */}
                   <TextField
                        margin="normal"
                        required
                        fullWidth
                        id="email"
                        label="Email"
                        name="email"
                        autoComplete="email"
                        autoFocus
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        disabled={isLoading}
                        error={isError && (localError.includes('email') || (!localError && authError))} // Подсветка при ошибке email или общей API ошибке
                        variant="outlined"
                    />
                    <TextField
                        margin="normal"
                        required
                        fullWidth
                        name="password"
                        label="Password"
                        type="password"
                        id="password"
                        autoComplete="new-password" // Подсказка для нового пароля
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        disabled={isLoading}
                        error={isError && (localError.includes('парол') || (!localError && authError))} // Подсветка при ошибке пароля или общей API ошибке
                        variant="outlined"
                    />
                     <TextField
                        margin="normal"
                        required
                        fullWidth
                        name="confirmPassword"
                        label="Confirm Password"
                        type="password"
                        id="confirmPassword"
                        autoComplete="new-password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        disabled={isLoading}
                        error={isError && localError.includes('Пароли не совпадают')} // Подсветка только если ошибка связана с несовпадением
                        variant="outlined"
                    />

                    {/* Отображение общей ошибки */}
                    {isError && (
                      <Alert severity="error" sx={{ width: '100%', mt: 1, mb: 1 }}>
                          {errorMessage}
                      </Alert>
                    )}

                    <Button
                        type="submit"
                        fullWidth
                        variant="contained" // Стиль кнопки
                        // color="secondary" // Можно использовать другой цвет для регистрации
                        sx={{ mt: 3, mb: 2, position: 'relative' }}
                        disabled={isLoading}
                    >
                        {isLoading ? 'Register' : 'Register'}
                        {isLoading && (
                            <CircularProgress
                                size={24}
                                sx={{
                                    position: 'absolute',
                                    top: '50%',
                                    left: '50%',
                                    marginTop: '-12px',
                                    marginLeft: '-12px',
                                }}
                            />
                        )}
                    </Button>

                    {/* Ссылка на вход */}
                    {/* <Box sx={{ textAlign: 'center' }}>
                        <Link component={RouterLink} to="/login" variant="body2">
                            {"Уже есть аккаунт? Войти"}
                        </Link>
                    </Box> */}
                </Box>
            </Box>
        </Container>
    );
}

export default RegisterForm;