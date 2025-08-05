import { useEffect, useState } from 'react';

function App() {
  const [transactions, setTransactions] = useState([]);

  useEffect(() => {
    fetch('http://localhost:8000/transactions')
      .then(res => res.json())
      .then(data => setTransactions(data));
  }, []);

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold">Fraud Alerts</h1>
      <ul>
        {transactions.map(tx => (
          <li key={tx.id}>
            {tx.id}: ${tx.amount} - {tx.fraud ? "🚨 FRAUD" : "✅ OK"}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
