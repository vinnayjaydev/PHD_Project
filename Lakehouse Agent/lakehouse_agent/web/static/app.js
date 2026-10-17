// Lakehouse Agent Frontend Application Logic

let currentAllocationChart = null;
let targetAllocationChart = null;
let currentInvestorId = "INV001";

document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  initCharts();
  loadInvestorAnalytics(currentInvestorId);
});

// Tab Navigation
function switchTab(tabId) {
  const tabs = ['advisory', 'lakehouse', 'scenarios', 'architecture'];
  tabs.forEach(t => {
    const panel = document.getElementById(`panel-${t}`);
    const tabBtn = document.getElementById(`tab-${t}`);
    if (t === tabId) {
      panel.classList.remove('hidden');
      tabBtn.classList.add('border-cyan-400', 'text-cyan-400');
      tabBtn.classList.remove('border-transparent', 'text-slate-400');
    } else {
      panel.classList.add('hidden');
      tabBtn.classList.remove('border-cyan-400', 'text-cyan-400');
      tabBtn.classList.add('border-transparent', 'text-slate-400');
    }
  });
  lucide.createIcons();
}

// Chart Initializations
function initCharts() {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: { color: '#94a3b8', font: { size: 11 } }
      }
    },
    cutout: '65%'
  };

  const ctxCurr = document.getElementById('chart-current-allocation').getContext('2d');
  currentAllocationChart = new Chart(ctxCurr, {
    type: 'doughnut',
    data: {
      labels: ['Equities', 'Fixed Income', 'Real Estate', 'Commodities', 'Cash'],
      datasets: [{
        data: [85, 0, 0, 0, 15],
        backgroundColor: ['#06b6d4', '#3b82f6', '#8b5cf6', '#f59e0b', '#64748b'],
        borderWidth: 2,
        borderColor: '#111827'
      }]
    },
    options: chartOptions
  });

  const ctxTgt = document.getElementById('chart-target-allocation').getContext('2d');
  targetAllocationChart = new Chart(ctxTgt, {
    type: 'doughnut',
    data: {
      labels: ['Equities', 'Fixed Income', 'Real Estate', 'Commodities', 'Cash'],
      datasets: [{
        data: [60, 25, 5, 5, 5],
        backgroundColor: ['#06b6d4', '#3b82f6', '#8b5cf6', '#f59e0b', '#64748b'],
        borderWidth: 2,
        borderColor: '#111827'
      }]
    },
    options: chartOptions
  });
}

// Fetch Investor Details
async function onInvestorChange() {
  const select = document.getElementById('select-investor');
  currentInvestorId = select.value;
  await loadInvestorAnalytics(currentInvestorId);
}

async function loadInvestorAnalytics(investorId) {
  try {
    const res = await fetch(`/api/investor/${investorId}/analytics`);
    const data = await res.json();
    const a = data.analytics;

    // Update Header Badges
    document.getElementById('badge-networth').innerText = `$${a.total_portfolio_value.toLocaleString()}`;
    document.getElementById('badge-risk-goal').innerText = `${a.stated_risk_tolerance} | ${a.primary_goal}`;

    document.getElementById('curr-beta').innerText = a.portfolio_beta;
    document.getElementById('curr-yield').innerText = `${a.portfolio_dividend_yield_pct}%`;
    document.getElementById('curr-div').innerText = `${a.diversification_score}/100`;
    document.getElementById('curr-drift').innerText = a.risk_drift_status;

    // Update Current Donut Chart
    currentAllocationChart.data.datasets[0].data = [
      a.equity_pct, a.bond_pct, a.real_estate_pct, a.commodities_pct, a.cash_pct
    ];
    currentAllocationChart.update();

  } catch (err) {
    console.error("Error loading investor analytics:", err);
  }
}

// Run Multi-Agent Recommendation Pipeline
async function runAgenticPipeline() {
  const btn = document.getElementById('btn-run-agents');
  const originalText = btn.innerHTML;
  btn.innerHTML = `<span class="animate-spin inline-block mr-2">&#9696;</span> Orchestrating Agents...`;
  btn.disabled = true;

  try {
    const res = await fetch('/api/recommendation/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ investor_id: currentInvestorId })
    });
    const report = await res.json();

    // 1. Update Title and Header Badges
    document.getElementById('rec-id-badge').innerText = `ID: ${report.recommendation_id} (${report.duration_seconds}s)`;
    document.getElementById('executive-summary-text').innerText = report.executive_summary;

    // 2. Compliance Badge
    const compBadge = document.getElementById('compliance-status-badge');
    if (report.compliance.is_compliant) {
      compBadge.className = "px-3 py-1.5 rounded-lg text-xs font-bold badge-approved flex items-center space-x-1.5";
      compBadge.innerHTML = `<i data-lucide="check-circle" class="w-4 h-4"></i><span>FINRA & Reg BI Approved</span>`;
    } else {
      compBadge.className = "px-3 py-1.5 rounded-lg text-xs font-bold badge-warning flex items-center space-x-1.5";
      compBadge.innerHTML = `<i data-lucide="alert-triangle" class="w-4 h-4"></i><span>Fiduciary Corrections Applied</span>`;
    }

    // 3. Update Target Chart
    const tgt = report.target_allocation;
    targetAllocationChart.data.datasets[0].data = [
      tgt.Equities || 0,
      tgt['Fixed Income'] || 0,
      tgt['Real Estate'] || 0,
      tgt.Commodities || 0,
      tgt.Cash || 0
    ];
    targetAllocationChart.update();

    // 4. Update Rebalancing Trade Orders Table
    const tbody = document.getElementById('orders-tbody');
    tbody.innerHTML = '';
    if (report.orders && report.orders.length > 0) {
      report.orders.forEach(o => {
        const actionClass = o.action === 'BUY' ? 'text-emerald-400 bg-emerald-950/60 border-emerald-800' : 'text-rose-400 bg-rose-950/60 border-rose-800';
        const row = document.createElement('tr');
        row.innerHTML = `
          <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[11px] font-bold border ${actionClass}">${o.action}</span></td>
          <td class="py-3 px-4 font-semibold text-slate-100">${o.ticker} <span class="text-slate-500 font-normal">(${o.asset_name})</span></td>
          <td class="py-3 px-4 text-slate-400">${o.asset_class}</td>
          <td class="py-3 px-4 font-mono">${o.shares}</td>
          <td class="py-3 px-4 font-mono">$${o.price.toFixed(2)}</td>
          <td class="py-3 px-4 font-mono font-semibold text-slate-200">$${o.estimated_amount.toLocaleString()}</td>
          <td class="py-3 px-4 text-slate-300 max-w-xs">${o.reason}</td>
        `;
        tbody.appendChild(row);
      });
    } else {
      tbody.innerHTML = `<tr><td colspan="7" class="py-4 text-center text-slate-500">Portfolio is already optimally balanced. No trades required.</td></tr>`;
    }

    // 5. Update Agent Cognitive Traces
    const traces = report.agent_traces;
    if (traces.user_profiler && traces.user_profiler.length > 0) {
      const p = traces.user_profiler.find(t => t.step === 'llm_synthesis_complete');
      document.getElementById('trace-profiler-summary').innerText = p ? p.details.preview : "Profile synthesized.";
    }
    if (traces.lakehouse_analytics && traces.lakehouse_analytics.length > 0) {
      const a = traces.lakehouse_analytics.find(t => t.step === 'llm_diagnostic_complete');
      document.getElementById('trace-analyst-summary').innerText = a ? a.details.preview : "Portfolio analytics evaluated.";
    }
    if (traces.market_intelligence && traces.market_intelligence.length > 0) {
      const m = traces.market_intelligence.find(t => t.step === 'llm_synthesis_complete');
      document.getElementById('trace-market-summary').innerText = m ? m.details.preview : "SEC & Macro RAG grounded.";
    }
    if (traces.portfolio_optimizer && traces.portfolio_optimizer.length > 0) {
      const opt = traces.portfolio_optimizer.find(t => t.step === 'llm_optimizer_complete');
      document.getElementById('trace-optimizer-summary').innerText = opt ? opt.details.preview : "MPT allocation solved.";
    }
    if (traces.compliance_guard && traces.compliance_guard.length > 0) {
      const c = traces.compliance_guard.find(t => t.step === 'llm_compliance_complete');
      document.getElementById('trace-compliance-summary').innerText = c ? c.details.preview : "Reg BI suitability validated.";
    }

    // 6. Update Citations List
    const citContainer = document.getElementById('citations-list');
    citContainer.innerHTML = '';
    if (report.rag_citations && report.rag_citations.length > 0) {
      report.rag_citations.forEach(c => {
        const item = document.createElement('div');
        item.className = "p-2.5 rounded bg-slate-900 border border-slate-800 text-xs text-indigo-300 flex items-center space-x-2";
        item.innerHTML = `<i data-lucide="bookmark" class="w-3.5 h-3.5 text-indigo-400"></i><span>${c}</span>`;
        citContainer.appendChild(item);
      });
    }

    lucide.createIcons();

  } catch (err) {
    alert("Agent pipeline error: " + err.message);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}

// SQL Console Handlers
function loadTablePreset(query) {
  document.getElementById('sql-query-input').value = query;
  executeCustomSQL();
}

async function executeCustomSQL() {
  const query = document.getElementById('sql-query-input').value.trim();
  const thead = document.getElementById('sql-thead');
  const tbody = document.getElementById('sql-tbody');
  const countLabel = document.getElementById('query-results-count');

  tbody.innerHTML = `<tr><td class="p-3 text-cyan-400">Executing DuckDB query...</td></tr>`;

  try {
    const res = await fetch('/api/lakehouse/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const result = await res.json();

    if (result.data && result.data.length > 0) {
      const cols = Object.keys(result.data[0]);
      countLabel.innerText = `Results (${result.row_count} rows returned)`;

      // Build Header
      thead.innerHTML = `<tr>${cols.map(c => `<th class="p-2.5 text-slate-300 font-semibold bg-slate-900">${c}</th>`).join('')}</tr>`;

      // Build Rows
      tbody.innerHTML = result.data.map(row => {
        return `<tr class="hover:bg-slate-900/50">${cols.map(c => `<td class="p-2.5 text-slate-300">${row[c] !== null ? row[c] : ''}</td>`).join('')}</tr>`;
      }).join('');
    } else {
      countLabel.innerText = `Results (0 rows)`;
      thead.innerHTML = `<tr><th class="p-2.5">Status</th></tr>`;
      tbody.innerHTML = `<tr><td class="p-2.5 text-slate-500">Query returned 0 rows or empty set.</td></tr>`;
    }
  } catch (err) {
    countLabel.innerText = `Query Error`;
    tbody.innerHTML = `<tr><td class="p-3 text-rose-400">Error: ${err.message}</td></tr>`;
  }
}

// Scenario Stress Test Handlers
async function runScenarioSimulation(scenarioKey) {
  const card = document.getElementById('stress-results-card');
  card.classList.remove('hidden');

  try {
    const res = await fetch('/api/scenario/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ investor_id: currentInvestorId, scenario: scenarioKey })
    });
    const data = await res.json();

    document.getElementById('stress-scenario-name').innerText = data.scenario;
    document.getElementById('stress-scenario-desc').innerText = data.description;
    document.getElementById('stress-alpha-val').innerText = `+${data.protection_alpha_pct}% Capital Preserved`;

    document.getElementById('stress-curr-val').innerText = `$${data.current_portfolio.simulated_value.toLocaleString()}`;
    document.getElementById('stress-curr-drawdown').innerText = `${data.current_portfolio.drawdown_pct}%`;

    document.getElementById('stress-target-val').innerText = `$${data.optimized_target_portfolio.simulated_value.toLocaleString()}`;
    document.getElementById('stress-target-drawdown').innerText = `${data.optimized_target_portfolio.drawdown_pct}%`;

  } catch (err) {
    alert("Simulation error: " + err.message);
  }
}

// Refresh Lakehouse Pipeline
async function refreshLakehousePipeline() {
  const btn = document.getElementById('btn-refresh-pipeline');
  btn.innerHTML = `<span class="animate-spin mr-1">&#9696;</span> Refreshing...`;
  btn.disabled = true;

  try {
    const res = await fetch('/api/pipeline/initialize', { method: 'POST' });
    const data = await res.json();
    alert("Lakehouse refreshed! " + data.message);
    await loadInvestorAnalytics(currentInvestorId);
  } catch (err) {
    alert("Failed to refresh lakehouse: " + err.message);
  } finally {
    btn.innerHTML = `<i data-lucide="refresh-cw" class="w-3.5 h-3.5 mr-1.5"></i><span>Refresh Data</span>`;
    btn.disabled = false;
    lucide.createIcons();
  }
}
