import { useState, useContext, useEffect } from 'react';
import { AuthContext } from '../context/AuthContext';

const API_BASE_URL = import.meta.env.VITE_API_URL;

export default function Prediction() {
  const { token } = useContext(AuthContext);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [formData, setFormData] = useState({
    age: 50, sex: 'Male', cp: 'typical angina', trestbps: 120,
    chol: 200, fbs: 'False', restecg: 'normal', thalch: 150,
    exang: 'False', oldpeak: 1.0, slope: 'upsloping'
  });

  useEffect(() => {
    if (token) {
      fetch(`${API_BASE_URL}/history`, { headers: { 'X-Token': token } })
        .then(res => res.json())
        .then(data => setHistory(data))
        .catch(console.error);
    }
  }, [token]);

  const handleChange = (e) => setFormData({ ...formData, [e.target.name]: e.target.value });

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Token': token || '' },
        body: JSON.stringify({
          ...formData,
          age: Number(formData.age),
          trestbps: Number(formData.trestbps),
          chol: Number(formData.chol),
          thalch: Number(formData.thalch),
          oldpeak: Number(formData.oldpeak)
        })
      });

      if (!res.ok) throw new Error('Помилка сервера під час аналізу');

      const data = await res.json();
      setResult(data);
      if (token) setHistory(prev => [data, ...prev]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const inputClass = "w-full border border-gray-300 rounded-lg p-2.5 focus:ring-red-500 focus:border-red-500 shadow-sm bg-gray-50 text-gray-900";
  const labelClass = "block text-sm font-semibold text-gray-700 mb-1";

  return (
    <div className="flex gap-8 flex-col lg:flex-row">
      <div className="flex-1">

        <div className="mb-6">
          <h1 className="text-3xl font-extrabold text-gray-900">🫀 Patient Data Analysis</h1>
          <p className="text-gray-500 mt-2">Fill in all 11 clinical indicators to run the ML forecast.</p>
        </div>

        {error && <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-4 mb-6 rounded">{error}</div>}

        <form onSubmit={handlePredict} className="bg-white p-8 rounded-2xl border border-gray-100 shadow-sm mb-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

            <div className="space-y-4">
              <div>
                <label className={labelClass}>Age</label>
                <input type="number" name="age" value={formData.age} onChange={handleChange} className={inputClass} />
              </div>
              <div>
                <label className={labelClass}>Sex</label>
                <select name="sex" value={formData.sex} onChange={handleChange} className={inputClass}>
                  <option>Male</option><option>Female</option>
                </select>
              </div>
              <div>
                <label className={labelClass}>Chest pain type</label>
                <select name="cp" value={formData.cp} onChange={handleChange} className={inputClass}>
                  <option>typical angina</option><option>atypical angina</option><option>non-anginal</option><option>asymptomatic</option>
                </select>
              </div>
              <div>
                <label className={labelClass}>Resting BP (mm Hg)</label>
                <input type="number" name="trestbps" value={formData.trestbps} onChange={handleChange} className={inputClass} />
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className={labelClass}>Cholesterol</label>
                <input type="number" name="chol" value={formData.chol} onChange={handleChange} className={inputClass} />
              </div>
              <div>
                <label className={labelClass}>Fasting sugar &gt; 120 mg/dl</label>
                <select name="fbs" value={formData.fbs} onChange={handleChange} className={inputClass}>
                  <option>True</option><option>False</option>
                </select>
              </div>
              <div>
                <label className={labelClass}>ECG Results</label>
                <select name="restecg" value={formData.restecg} onChange={handleChange} className={inputClass}>
                  <option>normal</option><option>st-t abnormality</option><option>lv hypertrophy</option>
                </select>
              </div>
              <div>
                <label className={labelClass}>Max Heart Rate</label>
                <input type="number" name="thalch" value={formData.thalch} onChange={handleChange} className={inputClass} />
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className={labelClass}>Exercise Angina</label>
                <select name="exang" value={formData.exang} onChange={handleChange} className={inputClass}>
                  <option>True</option><option>False</option>
                </select>
              </div>
              <div>
                <label className={labelClass}>ST depression</label>
                <input type="number" step="0.1" name="oldpeak" value={formData.oldpeak} onChange={handleChange} className={inputClass} />
              </div>
              <div>
                <label className={labelClass}>ST Slope</label>
                <select name="slope" value={formData.slope} onChange={handleChange} className={inputClass}>
                  <option>upsloping</option><option>flat</option><option>downsloping</option>
                </select>
              </div>
            </div>
          </div>

          <div className="mt-8 pt-6 border-t border-gray-100">
            <button type="submit" disabled={loading} className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-4 rounded-xl shadow-md transition-all transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed text-lg">
              {loading ? '⏳ Analyzing data...' : '🚀 Run Analysis'}
            </button>
          </div>
        </form>

        {result && (
          <div className="bg-white p-8 rounded-2xl border border-gray-100 shadow-sm animate-fade-in-up">
            <div className="flex items-center gap-4 mb-6">
              <h3 className="text-2xl font-bold text-gray-900">Analysis Result</h3>
              <span className={`px-4 py-1.5 rounded-full text-sm font-bold ${
                result.probability >= 70 ? 'bg-red-100 text-red-800' : 
                result.probability >= 45 ? 'bg-yellow-100 text-yellow-800' : 
                'bg-green-100 text-green-800'
              }`}>
                {result.probability >= 70 ? 'High Risk' : result.probability >= 45 ? 'Increased Risk' : 'Low Risk'}
              </span>
            </div>

            <div className="text-5xl font-extrabold mb-8 text-gray-900">
              {result.probability?.toFixed(1)}<span className="text-3xl text-gray-400">%</span>
            </div>

            <h4 className="text-lg font-bold mb-4 text-gray-700 border-t pt-6">Model Explanation (SHAP)</h4>
            {result.shap_plot ? (
              <div className="rounded-xl overflow-hidden border border-gray-200">
                <img src={`data:image/png;base64,${result.shap_plot}`} alt="SHAP Plot" className="w-full object-contain" />
              </div>
            ) : (
              <p className="text-gray-500 italic">Графік недоступний</p>
            )}
          </div>
        )}
      </div>

      <div className="w-full lg:w-80 shrink-0">
        <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm sticky top-24">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-gray-900">
            <span>📊</span> Check History
          </h3>

          {!token ? (
            <div className="bg-red-50 text-red-700 p-4 rounded-xl text-sm border border-red-100">
              Please <strong>log in</strong> to save and view your analysis history.
            </div>
          ) : history.length === 0 ? (
            <p className="text-sm text-gray-500 text-center py-4">Your history is empty.</p>
          ) : (
            <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-2 custom-scrollbar">
              {history.map((item, i) => {
                const dateStr = item.created_at ? new Date(item.created_at).toLocaleString([], {day: '2-digit', month: '2-digit', hour: '2-digit', minute:'2-digit'}) : `Check #${history.length - i}`;
                return (
                  <div
                    key={item.id || i}
                    onClick={() => {
                      setResult({ probability: item.probability, shap_plot: item.shap_image_base64 });
                      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                    }}
                    className="group p-3 border border-gray-100 rounded-xl cursor-pointer hover:bg-red-50 hover:border-red-200 transition-all flex justify-between items-center"
                  >
                    <span className="text-sm text-gray-500 group-hover:text-red-700 font-medium">🕒 {dateStr}</span>
                    <span className={`text-sm font-bold ${item.probability >= 70 ? 'text-red-600' : item.probability >= 45 ? 'text-yellow-600' : 'text-green-600'}`}>
                      {item.probability?.toFixed(1)}%
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}