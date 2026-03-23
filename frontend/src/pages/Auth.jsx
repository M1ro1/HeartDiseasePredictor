import { useState, useContext } from 'react';
import { AuthContext } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_URL;

export default function Auth() {
  const [isLogin, setIsLogin] = useState(true);
  const { login } = useContext(AuthContext);
  const navigate = useNavigate();
  const [formData, setFormData] = useState({ username: '', email: '', password: '' });
  const [error, setError] = useState('');

  const handleChange = (e) => setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    try {
      if (isLogin) {
        const params = new URLSearchParams();
        params.append('username', formData.username);
        params.append('password', formData.password);

        const res = await fetch(`${API_BASE_URL}/token`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: params
        });

        if (!res.ok) throw new Error('Wrong login or password');
        const data = await res.json();
        login({ username: formData.username }, data.access_token);
        navigate('/predict');
      } else {
        const res = await fetch(`${API_BASE_URL}/registration`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: formData.username, email: formData.email, password: formData.password })
        });
        if (!res.ok) throw new Error('Registration failed');
        alert('Success! Please log in.');
        setIsLogin(true);
      }
    } catch (err) { setError(err.message); }
  };

  return (
    <div className="max-w-md mx-auto bg-white p-8 rounded-xl shadow-sm border mt-10">
      <h2 className="text-2xl font-bold mb-6 text-center">{isLogin ? 'Login' : 'Registration'}</h2>
      {error && <div className="bg-red-100 text-red-700 p-3 mb-4 rounded">{error}</div>}
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <input name="username" placeholder="Username" required onChange={handleChange} className="border p-2 rounded" />
        {!isLogin && <input name="email" type="email" placeholder="Email" required onChange={handleChange} className="border p-2 rounded" />}
        <input name="password" type="password" placeholder="Password" required onChange={handleChange} className="border p-2 rounded" />
        <button type="submit" className="bg-red-500 text-white font-bold py-2 rounded mt-2">{isLogin ? 'Увійти' : 'Зареєструватись'}</button>
      </form>
      <button onClick={() => setIsLogin(!isLogin)} className="text-red-500 text-sm mt-4 w-full text-center">
        {isLogin ? "Don't have an account? Sign up" : "Already have an account? Log in"}
      </button>
    </div>
  );
}