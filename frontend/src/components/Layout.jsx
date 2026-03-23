import { useContext } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';

export default function Layout() {
  const { user, token, logout } = useContext(AuthContext);
  const location = useLocation();

  return (
    <div className="min-h-screen font-sans flex flex-col">
      <header className="bg-white border-b shadow-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 h-16 flex justify-between items-center">
          <Link to="/" className="text-xl font-bold text-red-500 flex items-center gap-2">
            <span>🫀</span> Heart AI
          </Link>
          <nav className="flex gap-4">
            <Link to="/" className={location.pathname === '/' ? 'text-red-600 font-bold' : 'text-gray-600'}>Home</Link>
            <Link to="/predict" className={location.pathname === '/predict' ? 'text-red-600 font-bold' : 'text-gray-600'}>Prediction</Link>
          </nav>
          <div>
            {token ? (
              <div className="flex items-center gap-4">
                <span className="font-bold">{user?.username}</span>
                <button onClick={logout} className="text-red-500 hover:text-red-700 font-semibold">Logout</button>
              </div>
            ) : (
              <Link to="/auth" className="bg-red-500 text-white px-4 py-2 rounded-md font-bold">Login</Link>
            )}
          </div>
        </div>
      </header>
      <main className="flex-1 w-full max-w-6xl mx-auto p-4 sm:p-6 lg:p-8">
        <Outlet />
      </main>
    </div>
  );
}