import { Link } from 'react-router-dom';

const BeakerIcon = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v1.242c0 .289-.246.522-.52.522-.51 0-.927.424-.927.934a.925.925 0 0 0 .927.935h2.25a.927.927 0 0 0 .927-.935c0-.51-.417-.934-.927-.934a.52.52 0 0 1-.52-.522V3.104m-2.25 0a.75.75 0 0 1 .75-.75h4.5a.75.75 0 0 1 .75.75m-6.75 0v3.75c0 .324.032.645.093.959M9 10.5h6" /></svg>;
const BrainIcon = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 0 0 1.5-.189m-1.5.189a6.01 6.01 0 0 1-1.5-.189m3.75 7.478a12.06 12.06 0 0 1-4.5 0m3.75 2.383a14.406 14.406 0 0 1-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 1 0-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" /></svg>;
const ChartBarIcon = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" /></svg>;
const CheckCircleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" /></svg>;

export default function Home() {
  const features = [
    { icon: <BeakerIcon />, title: 'Advanced ML Model', description: 'Our algorithm analysis 11 key clinical markers using state-of-the-art ensemble modeling.' },
    { icon: <BrainIcon />, title: 'Explainable AI (XAI)', description: 'We don’t just give you a number. Integrated SHAP graphs show exactly how each factor influences your risk.' },
    { icon: <ChartBarIcon />, title: 'Risk Probability Score', description: 'Receive a precise percentage indicating your statistical risk group, from low to high.' },
    { icon: <CheckCircleIcon />, title: 'Valid Medical Indicators', description: 'Analysis is based on proven indicators like ECG results, chest pain type, cholesterol, and more.' },
  ];

  const stats = [
    { id: 1, name: 'Clinical Parameters Analyzed', value: '11' },
    { id: 2, name: 'Analysis Speed', value: '< 3 sec' },
    { id: 3, name: 'Data Security', value: '100%' },
  ];

  return (
    <div className="relative overflow-hidden bg-gray-50 flex flex-col min-h-screen">

      <div className="absolute inset-0 -z-10 transform-gpu overflow-hidden blur-3xl" aria-hidden="true">
        <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#ff80b0] to-[#ff4b4b] opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" style={{ clipPath: 'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)' }} />
      </div>

      <div className="flex-grow">
        <div className="py-16 sm:py-20">
          <div className="mx-auto max-w-7xl px-6 lg:px-8 text-center">
            <div className="mx-auto max-w-3xl">
              <h1 className="text-5xl font-extrabold tracking-tight text-gray-900 sm:text-6xl bg-gradient-to-r from-red-600 to-red-400 bg-clip-text text-transparent pb-2">
                🫀 AI Heart Disease Risk Analyzer
              </h1>
              <p className="mt-6 text-xl leading-8 text-gray-600">
                A reliable and comprehensive tool for cardiovascular risk assessment powered by advanced machine learning models.
              </p>
              <div className="mt-10 flex items-center justify-center gap-x-6">
                <Link to="/predict" className="rounded-xl bg-red-600 px-8 py-4 text-lg font-bold text-white shadow-lg hover:bg-red-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-600 transition-all transform hover:-translate-y-1">
                  🚀 Go to Forecast Panel
                </Link>
                <Link to="/auth" className="text-lg font-semibold leading-6 text-gray-900 group">
                  Sign Up Now <span aria-hidden="true" className="inline-block transition-transform group-hover:translate-x-1">→</span>
                </Link>
              </div>
            </div>
          </div>
        </div>

        <div className="mx-auto max-w-7xl px-6 lg:px-8 pb-12">
          <dl className="grid grid-cols-1 gap-x-8 gap-y-16 text-center lg:grid-cols-3 bg-white p-8 rounded-3xl shadow-sm border border-gray-100">
            {stats.map((stat) => (
              <div key={stat.id} className="mx-auto flex max-w-xs flex-col gap-y-4">
                <dt className="text-base leading-7 text-gray-600">{stat.name}</dt>
                <dd className="order-first text-3xl font-semibold tracking-tight text-gray-900 sm:text-5xl">
                  {stat.value}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="mx-auto max-w-7xl px-6 lg:px-8 py-12">
          <div className="mx-auto max-w-2xl lg:text-center mb-16">
            <h2 className="text-base font-semibold leading-7 text-red-600">Smart Technology</h2>
            <p className="mt-2 text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl">Analyze. Explain. Understand.</p>
          </div>
          <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
            <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-4">
              {features.map((feature, index) => (
                <div key={index} className="relative pl-16 bg-white p-6 rounded-3xl shadow-sm border border-gray-100 hover:shadow-lg transition-shadow">
                  <dt className="text-base font-semibold leading-7 text-gray-900">
                    <div className="absolute left-6 top-6 flex h-10 w-10 items-center justify-center rounded-lg bg-red-100 text-red-600">{feature.icon}</div>
                    {feature.title}
                  </dt>
                  <dd className="mt-2 text-base leading-7 text-gray-600">{feature.description}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>

        <div className="mx-auto max-w-7xl px-6 lg:px-8 py-16">
          <div className="relative isolate overflow-hidden bg-red-600 px-6 py-16 text-center shadow-2xl rounded-3xl sm:px-16">
            <h2 className="mx-auto max-w-2xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
              Ready to check your metrics?
            </h2>
            <p className="mx-auto mt-6 max-w-xl text-lg leading-8 text-red-100">
              Join our platform today to get a detailed analysis of your cardiovascular health risk factors.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link to="/predict" className="rounded-xl bg-white px-8 py-3.5 text-lg font-bold text-red-600 shadow-sm hover:bg-gray-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white transition-all transform hover:scale-105">
                Start Analysis Now
              </Link>
            </div>
            <svg viewBox="0 0 1024 1024" className="absolute left-1/2 top-1/2 -z-10 h-[64rem] w-[64rem] -translate-x-1/2 [mask-image:radial-gradient(closest-side,white,transparent)]" aria-hidden="true">
              <circle cx={512} cy={512} r={512} fill="url(#827591b1-ce8c-4110-b064-7cb85a0b1217)" fillOpacity="0.7" />
              <defs><radialGradient id="827591b1-ce8c-4110-b064-7cb85a0b1217"><stop stopColor="#ffb3c6" /><stop offset={1} stopColor="#ff4b4b" /></radialGradient></defs>
            </svg>
          </div>
        </div>

        <div className="mx-auto max-w-7xl px-6 lg:px-8 pb-16 grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100">
            <h3 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2"><span>🔬</span> About the Project</h3>
            <p className="text-gray-700 mb-4 leading-relaxed">This application is designed to support early detection of potential heart disease risks. Our tool utilizes ensemble learning models trained on robust datasets to ensure maximum reliable prediction.</p>
          </div>
          <div className="bg-yellow-50 p-8 rounded-3xl border border-yellow-200">
            <h3 className="text-2xl font-bold text-yellow-900 mb-4 flex items-center gap-2"><span>⚠️</span> Medical Disclaimer</h3>
            <p className="text-yellow-800 leading-relaxed">This app is for <strong>educational and informational purposes only</strong>. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult your doctor.</p>
          </div>
        </div>

      </div>

      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="mx-auto max-w-7xl px-6 py-8 md:flex md:items-center md:justify-between lg:px-8">
          <div className="flex justify-center space-x-6 md:order-2 text-sm text-gray-500">
            <span className="hover:text-red-600 cursor-pointer transition-colors">Privacy Policy</span>
            <span className="hover:text-red-600 cursor-pointer transition-colors">Terms of Service</span>
            <span className="hover:text-red-600 cursor-pointer transition-colors">Contact</span>
          </div>
          <div className="mt-8 md:order-1 md:mt-0">
            <p className="text-center text-sm leading-5 text-gray-500">
              &copy; {new Date().getFullYear()} Heart AI Project. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}