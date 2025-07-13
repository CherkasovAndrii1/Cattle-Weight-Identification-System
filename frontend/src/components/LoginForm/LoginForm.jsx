import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext'; // Путь к вашему контексту
import { useNavigate, Link as RouterLink } from 'react-router-dom'; // Используем Link из роутера

// MUI Компоненты
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import Alert from '@mui/material/Alert';
import Link from '@mui/material/Link'; // MUI Link для стилизации
import CircularProgress from '@mui/material/CircularProgress'; // Для индикатора загрузки

// (Опционально) Иконки, если нужны
// import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
// import Avatar from '@mui/material/Avatar';

function LoginForm() {
    const { login, isLoading, authError } = useAuth();
    const navigate = useNavigate();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [localError, setLocalError] = useState('');

    const handleSubmit = async (event) => {
        event.preventDefault();
        setLocalError('');

        if (!email || !password) {
            setLocalError('Please enter email and password');
            return;
        }
        // Простая валидация email (можно улучшить)
        if (!/\S+@\S+\.\S+/.test(email)) {
            setLocalError('Please enter correct email');
            return;
        }

        const success = await login({ email, password });
        if (success) {
            console.log("Login successful, navigating to home...");
            navigate('/'); 
        }
    };

    const isError = Boolean(localError || authError);
    const errorMessage = localError || authError || '';

    return (
        <Container component="main" maxWidth="xs"> {/* Ограничивает ширину и центрирует */}
            <Box
                sx={{
                    marginTop: 8, // Отступ сверху
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    bgcolor: 'background.paper', // Используем цвет фона из темы
                    padding: 4, // Внутренние отступы
                    borderRadius: 2, // Скругление углов
                    boxShadow: 3, // Стандартная тень MUI
                }}
            >
                {/* (Опционально) Иконка/Аватар */}
                {/* <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
                    <LockOutlinedIcon />
                 </Avatar> */}
                <Typography component="h1" variant="h5">
                    Log in
                </Typography>
                <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 1, width: '100%' }}>
                    <TextField
                        margin="normal" // Добавляет стандартные отступы
                        required // HTML5 атрибут
                        fullWidth // Занимает всю ширину
                        id="email"
                        label="Email" // Лейбл внутри или сверху поля
                        name="email"
                        autoComplete="email"
                        autoFocus // Фокус на этом поле при загрузке
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        disabled={isLoading}
                        error={isError && (localError.includes('email') || !localError)} // Подсвечиваем поле при ошибке email или общей
                        // helperText={isError && (localError.includes('email') || !localError) ? errorMessage : ''} // Можно вывести текст под полем
                        variant="outlined" // Стиль поля (outlined, filled, standard)
                    />
                    <TextField
                        margin="normal"
                        required
                        fullWidth
                        name="password"
                        label="password"
                        type="password"
                        id="password"
                        autoComplete="current-password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        disabled={isLoading}
                        error={isError && (localError.includes('парол') || !localError)} // Подсвечиваем при ошибке пароля или общей
                        // helperText={isError && (localError.includes('парол') || !localError) ? errorMessage : ''}
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
                        variant="contained" // Основной стиль кнопки MUI
                        sx={{ mt: 3, mb: 2, position: 'relative' }} // Отступы сверху/снизу
                        disabled={isLoading}
                    >
                        {isLoading ? 'Sign in' : 'Sign in'}
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

                    {/* Ссылка на регистрацию */}
                    {/* <Box sx={{ textAlign: 'center' }}>
                         <Link component={RouterLink} to="/register" variant="body2">
                            {"Нет аккаунта? Зарегистрироваться"}
                        </Link>
                    </Box> */}
                </Box>
            </Box>
        </Container>
    );
}

export default LoginForm;