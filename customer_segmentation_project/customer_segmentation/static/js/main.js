/* ═══════════════════════════════════════════════════════════
   CustomerIQ · Main JavaScript
═══════════════════════════════════════════════════════════ */

const COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
];

const SEGMENT_NAMES = [
  'High Spenders', 'Cautious Savers', 'Budget Shoppers',
  'Premium Loyalists', 'Mid-Range Explorers', 'Value Seekers',
  'Impulse Buyers', 'Consistent Shoppers', 'Occasional Visitors', 'VIP Clients'
];

const BG_COLOR   = '#0e1420';
const GRID_COLOR = 'rgba(255,255,255,0.05)';
const FONT_COLOR = '#7a8096';

let currentData = null;
let usingUpload = false;

const plotlyLayout = (extra = {}) => ({
  paper_bgcolor: BG_COLOR,
  plot_bgcolor:  BG_COLOR,
  font: { family: "'DM Sans', sans-serif", color: FONT_COLOR, size: 12 },
  margin: { t: 10, r: 20, b: 50, l: 50 },
  xaxis: {
    gridcolor: GRID_COLOR, zerolinecolor: GRID_COLOR,
    tickfont: { color: FONT_COLOR }, title: { font: { color: FONT_COLOR } }
  },
  yaxis: {
    gridcolor: GRID_COLOR, zerolinecolor: GRID_COLOR,
    tickfont: { color: FONT_COLOR }, title: { font: { color: FONT_COLOR } }
  },
  legend: {
    bgcolor: 'rgba(14,20,32,0.8)',
    bordercolor: 'rgba(255,255,255,0.08)',
    borderwidth: 1,
    font: { color: FONT_COLOR, size: 11 }
  },
  ...extra
});

const plotlyConfig = {
  displayModeBar: false,
  responsive: true
};

// ── Init ──────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  useSampleData();
});

// ── Data Source ───────────────────────────────────────────────
function useSampleData() {
  usingUpload = false;
  document.getElementById('btnSample').classList.add('active');
  document.getElementById('btnUpload').classList.remove('active');
  document.getElementById('uploadInfo').style.display = 'none';
  fetch('/api/sample_data')
    .then(r => r.json())
    .then(info => {
      updateFeatureChecks(info.numeric_columns);
    });
}

function handleUpload(input) {
  const file = input.files[0];
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  fetch('/api/upload', { method: 'POST', body: form })
    .then(r => r.json())
    .then(info => {
      if (info.error) { alert(info.error); return; }
      usingUpload = true;
      document.getElementById('btnUpload').classList.add('active');
      document.getElementById('btnSample').classList.remove('active');
      document.getElementById('uploadInfo').style.display = 'flex';
      document.getElementById('uploadName').textContent = file.name;
      document.getElementById('uploadRows').textContent = `${info.rows} rows`;
      updateFeatureChecks(info.numeric_columns);
    });
}

function updateFeatureChecks(numericCols) {
  const container = document.getElementById('featureChecks');
  const defaults = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age'];
  const cols = numericCols.length > 0 ? numericCols : defaults;
  container.innerHTML = cols.map((col, i) => `
    <label class="check-item">
      <input type="checkbox" value="${col}" ${i < 2 ? 'checked' : ''} class="feat-check"/>
      ${col}
    </label>
  `).join('');
}

// ── Slider ────────────────────────────────────────────────────
function updateK(val) {
  document.getElementById('kDisplay').textContent = val;
}

// ── Run Analysis ──────────────────────────────────────────────
async function runAnalysis() {
  const k = parseInt(document.getElementById('kSlider').value);
  const features = [...document.querySelectorAll('.feat-check:checked')].map(c => c.value);

  if (features.length < 1) {
    alert('Please select at least one feature.');
    return;
  }

  showLoading();

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_clusters: k, features })
    });
    const data = await res.json();
    currentData = data;

    await animateLoading();
    hideLoading();

    renderAll(data);
  } catch (err) {
    hideLoading();
    alert('Analysis failed: ' + err.message);
  }
}

// ── Render All ────────────────────────────────────────────────
function renderAll(data) {
  updateStats(data);
  renderScatter(data);
  renderElbow(data);
  renderPie(data);
  renderTable(data);
  renderLegend(data);
}

function updateStats(data) {
  animateNum('statCustomers', data.total_customers);
  animateNum('statClusters', data.n_clusters);
  document.getElementById('statInertia').textContent = data.inertia.toLocaleString();
  document.getElementById('scatterBadge').textContent = `K = ${data.n_clusters}`;
}

function animateNum(id, target) {
  const el = document.getElementById(id);
  let start = 0;
  const step = Math.ceil(target / 30);
  const timer = setInterval(() => {
    start = Math.min(start + step, target);
    el.textContent = start.toLocaleString();
    if (start >= target) clearInterval(timer);
  }, 30);
}

// ── Scatter Plot ──────────────────────────────────────────────
function renderScatter(data) {
  const traces = [];
  for (let c = 0; c < data.n_clusters; c++) {
    const pts = data.scatter.filter(p => p.cluster === c);
    traces.push({
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      mode: 'markers',
      type: 'scatter',
      name: SEGMENT_NAMES[c] || `Cluster ${c}`,
      marker: {
        color: COLORS[c % COLORS.length],
        size: 9,
        opacity: 0.85,
        line: { color: 'rgba(0,0,0,0.4)', width: 1 }
      },
      customdata: pts.map(p => [p.id, p.income, p.spending, p.age, p.gender]),
      hovertemplate:
        '<b>Customer #%{customdata[0]}</b><br>' +
        'Income: $%{customdata[1]}k<br>' +
        'Spending: %{customdata[2]}<br>' +
        'Age: %{customdata[3]}<br>' +
        'Gender: %{customdata[4]}<extra></extra>'
    });
  }

  const layout = plotlyLayout({
    xaxis: { ...plotlyLayout().xaxis, title: { text: data.x_label, font: { color: FONT_COLOR } } },
    yaxis: { ...plotlyLayout().yaxis, title: { text: data.y_label, font: { color: FONT_COLOR } } },
    margin: { t: 10, r: 20, b: 60, l: 60 },
    hovermode: 'closest',
    legend: { ...plotlyLayout().legend, orientation: 'h', y: -0.15 }
  });

  Plotly.newPlot('scatterPlot', traces, layout, plotlyConfig);
}

// ── Elbow Plot ────────────────────────────────────────────────
function renderElbow(data) {
  const ks = Array.from({ length: 9 }, (_, i) => i + 2);
  const currentK = data.n_clusters;

  const trace = {
    x: ks,
    y: data.inertias,
    mode: 'lines+markers',
    type: 'scatter',
    line: { color: '#5b8dee', width: 2.5 },
    marker: {
      color: ks.map(k => k === currentK ? '#FF6B6B' : '#5b8dee'),
      size: ks.map(k => k === currentK ? 12 : 7),
      line: { color: '#0e1420', width: 2 }
    },
    hovertemplate: 'K = %{x}<br>Inertia: %{y:.0f}<extra></extra>'
  };

  const layout = plotlyLayout({
    xaxis: { ...plotlyLayout().xaxis, title: { text: 'Number of Clusters (K)', font: { color: FONT_COLOR } }, dtick: 1 },
    yaxis: { ...plotlyLayout().yaxis, title: { text: 'Inertia', font: { color: FONT_COLOR } } },
    margin: { t: 10, r: 20, b: 55, l: 60 },
  });

  Plotly.newPlot('elbowPlot', [trace], layout, plotlyConfig);
}

// ── Pie Chart ─────────────────────────────────────────────────
function renderPie(data) {
  const labels = data.cluster_stats.map(s => SEGMENT_NAMES[s.cluster] || `Cluster ${s.cluster}`);
  const values = data.cluster_stats.map(s => s.count);
  const colors = data.cluster_stats.map(s => COLORS[s.cluster % COLORS.length]);

  const trace = {
    labels,
    values,
    type: 'pie',
    marker: { colors, line: { color: BG_COLOR, width: 2 } },
    textinfo: 'percent',
    textfont: { color: '#fff', size: 12, family: "'Syne', sans-serif" },
    hole: 0.45,
    hovertemplate: '<b>%{label}</b><br>%{value} customers (%{percent})<extra></extra>'
  };

  const layout = plotlyLayout({
    margin: { t: 10, r: 10, b: 10, l: 10 },
    legend: { ...plotlyLayout().legend, orientation: 'v', x: 1.05 },
    showlegend: false
  });

  Plotly.newPlot('piePlot', [trace], layout, plotlyConfig);
}

// ── Stats Table ───────────────────────────────────────────────
function renderTable(data) {
  const container = document.getElementById('statsTable');

  const incomeKey = data.features.find(f => f.toLowerCase().includes('income'));
  const spendKey  = data.features.find(f => f.toLowerCase().includes('spending'));

  const maxIncome  = Math.max(...data.cluster_stats.map(s => s[`avg_${incomeKey}`] || 0));
  const maxSpend   = Math.max(...data.cluster_stats.map(s => s[`avg_${spendKey}`] || 0));
  const maxCount   = Math.max(...data.cluster_stats.map(s => s.count));

  let html = `
    <table class="stats-table">
      <thead>
        <tr>
          <th>Segment</th>
          <th>Customers</th>
          ${incomeKey ? `<th>Avg Income</th>` : ''}
          ${spendKey  ? `<th>Avg Spending</th>` : ''}
          ${data.cluster_stats[0]?.avg_age ? `<th>Avg Age</th>` : ''}
          ${data.cluster_stats[0]?.dominant_gender ? `<th>Gender</th>` : ''}
        </tr>
      </thead>
      <tbody>
  `;

  data.cluster_stats.forEach(s => {
    const color = COLORS[s.cluster % COLORS.length];
    const name  = SEGMENT_NAMES[s.cluster] || `Cluster ${s.cluster}`;
    const income = incomeKey ? s[`avg_${incomeKey}`] : null;
    const spend  = spendKey  ? s[`avg_${spendKey}`]  : null;
    const countPct = Math.round((s.count / maxCount) * 100);

    html += `
      <tr>
        <td>
          <div class="cluster-pill" style="background:${color}22; border:1px solid ${color}44">
            <div class="cluster-dot" style="background:${color}"></div>
            <span style="color:${color}">${name}</span>
          </div>
        </td>
        <td>
          <div class="bar-wrap">
            <div class="mini-bar" style="background:${color};width:${countPct}px;max-width:80px"></div>
            <span>${s.count}</span>
          </div>
        </td>
        ${income !== null ? `
        <td>
          <div class="bar-wrap">
            <div class="mini-bar" style="background:#5b8dee;width:${Math.round((income/maxIncome)*80)}px"></div>
            <span>$${income}k</span>
          </div>
        </td>` : ''}
        ${spend !== null ? `
        <td>
          <div class="bar-wrap">
            <div class="mini-bar" style="background:#4ECDC4;width:${Math.round((spend/maxSpend)*80)}px"></div>
            <span>${spend}</span>
          </div>
        </td>` : ''}
        ${s.avg_age ? `<td>${s.avg_age} yrs</td>` : ''}
        ${s.dominant_gender ? `<td>${s.dominant_gender}</td>` : ''}
      </tr>
    `;
  });

  html += `</tbody></table>`;
  container.innerHTML = html;
}

// ── Legend ────────────────────────────────────────────────────
function renderLegend(data) {
  const legend = document.getElementById('clusterLegend');
  const items  = document.getElementById('legendItems');
  legend.style.display = 'block';

  items.innerHTML = data.cluster_stats.map(s => {
    const color = COLORS[s.cluster % COLORS.length];
    const name  = SEGMENT_NAMES[s.cluster] || `Cluster ${s.cluster}`;
    const pct   = Math.round((s.count / data.total_customers) * 100);
    return `
      <div class="legend-item">
        <div class="legend-dot" style="background:${color}"></div>
        <span class="legend-name">${name}</span>
        <span class="legend-count">${pct}%</span>
      </div>
    `;
  }).join('');
}

// ── Loading ───────────────────────────────────────────────────
function showLoading() {
  document.getElementById('loadingOverlay').style.display = 'flex';
  document.getElementById('step1').className = 'step active';
  document.getElementById('step2').className = 'step';
  document.getElementById('step3').className = 'step';
}

function hideLoading() {
  document.getElementById('loadingOverlay').style.display = 'none';
}

async function animateLoading() {
  await delay(400);
  document.getElementById('step1').className = 'step done';
  document.getElementById('step2').className = 'step active';
  await delay(400);
  document.getElementById('step2').className = 'step done';
  document.getElementById('step3').className = 'step active';
  await delay(300);
  document.getElementById('step3').className = 'step done';
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
