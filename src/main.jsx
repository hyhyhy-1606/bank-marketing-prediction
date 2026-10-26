
import React, { useMemo, useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import Papa from "papaparse";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip,
  CartesianGrid, PieChart, Pie, Cell, LineChart, Line
} from "recharts";
import {
  Activity, BarChart3, BrainCircuit, ChevronDown, CircleHelp,
  Database, FileUp, Filter, Gauge, LayoutDashboard, Search,
  ShieldCheck, Users, WalletCards, Download
} from "lucide-react";
import "./styles.css";

const DEMO = [
  { age: 30, job: "technician", marital: "married", education: "secondary", balance: 1200, housing: "yes", loan: "no", contact: "cellular", month: "may", campaign: 2, pdays: 999, previous: 0, poutcome: "unknown", y: "no" },
  { age: 42, job: "management", marital: "married", education: "tertiary", balance: 4200, housing: "no", loan: "no", contact: "cellular", month: "oct", campaign: 1, pdays: 999, previous: 0, poutcome: "unknown", y: "yes" },
  { age: 27, job: "student", marital: "single", education: "tertiary", balance: 650, housing: "no", loan: "no", contact: "cellular", month: "aug", campaign: 1, pdays: 999, previous: 0, poutcome: "unknown", y: "yes" },
  { age: 51, job: "blue-collar", marital: "married", education: "primary", balance: -120, housing: "yes", loan: "yes", contact: "telephone", month: "jun", campaign: 4, pdays: 999, previous: 0, poutcome: "unknown", y: "no" },
  { age: 36, job: "admin.", marital: "single", education: "secondary", balance: 900, housing: "yes", loan: "no", contact: "cellular", month: "may", campaign: 3, pdays: 12, previous: 2, poutcome: "success", y: "yes" },
  { age: 45, job: "management", marital: "divorced", education: "tertiary", balance: 2800, housing: "no", loan: "no", contact: "cellular", month: "nov", campaign: 1, pdays: 999, previous: 0, poutcome: "unknown", y: "no" },
  { age: 33, job: "services", marital: "married", education: "secondary", balance: 350, housing: "yes", loan: "no", contact: "unknown", month: "jul", campaign: 2, pdays: 999, previous: 0, poutcome: "unknown", y: "no" },
  { age: 29, job: "self-employed", marital: "single", education: "tertiary", balance: 1700, housing: "no", loan: "no", contact: "cellular", month: "sep", campaign: 1, pdays: 4, previous: 1, poutcome: "success", y: "yes" },
  { age: 58, job: "retired", marital: "married", education: "primary", balance: 5300, housing: "no", loan: "no", contact: "telephone", month: "mar", campaign: 1, pdays: 999, previous: 0, poutcome: "unknown", y: "yes" },
  { age: 39, job: "entrepreneur", marital: "married", education: "secondary", balance: 2200, housing: "yes", loan: "no", contact: "cellular", month: "may", campaign: 5, pdays: 999, previous: 0, poutcome: "unknown", y: "no" },
  { age: 24, job: "student", marital: "single", education: "tertiary", balance: 200, housing: "no", loan: "no", contact: "cellular", month: "dec", campaign: 1, pdays: 6, previous: 1, poutcome: "success", y: "yes" },
  { age: 48, job: "blue-collar", marital: "married", education: "secondary", balance: 780, housing: "yes", loan: "no", contact: "telephone", month: "jun", campaign: 3, pdays: 999, previous: 0, poutcome: "unknown", y: "no" }
];

const COLORS = ["#7c8cff", "#34d399", "#f59e0b", "#f472b6", "#38bdf8", "#a78bfa"];

const fmt = n => new Intl.NumberFormat("vi-VN").format(Number(n) || 0);
const pct = n => Number.isFinite(Number(n)) ? `${(Number(n) * 100).toFixed(1)}%` : "—";
const normYes = v => ["yes", "1", "true"].includes(String(v ?? "").trim().toLowerCase());
const safeNum = v => {
  if (v === null || v === undefined || v === "") return 0;
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
};

const REQUIRED_COLUMNS = [
  "age", "job", "marital", "education", "balance",
  "housing", "loan", "contact", "month", "campaign",
  "pdays", "previous", "poutcome", "y"
];

function normalizeMetrics(raw) {
  if (!raw || typeof raw !== "object") return null;

  const accuracy = Number(raw.accuracy);
  const precision = Number(raw.precision);
  const recall = Number(raw.recall);
  const f1 = Number(raw.f1);
  const roc_auc = Number(raw.roc_auc ?? raw.roc_auc_score);

  const validRate = v => Number.isFinite(v) && v >= 0 && v <= 1;

  if (![accuracy, precision, recall, f1, roc_auc].every(validRate)) {
    return null;
  }

  let tn = Number(raw.tn);
  let fp = Number(raw.fp);
  let fn = Number(raw.fn);
  let tp = Number(raw.tp);

  // Hỗ trợ confusion_matrix dạng [[TN, FP], [FN, TP]].
  const cm = raw.confusion_matrix;
  if (
    (!Number.isFinite(tn) || !Number.isFinite(fp) ||
      !Number.isFinite(fn) || !Number.isFinite(tp)) &&
    Array.isArray(cm) &&
    Array.isArray(cm[0]) &&
    Array.isArray(cm[1]) &&
    cm[0].length >= 2 &&
    cm[1].length >= 2
  ) {
    tn = Number(cm[0][0]);
    fp = Number(cm[0][1]);
    fn = Number(cm[1][0]);
    tp = Number(cm[1][1]);
  }

  const hasCM = [tn, fp, fn, tp].every(
    n => Number.isFinite(n) && n >= 0
  );

  const cmTotal = hasCM ? tn + fp + fn + tp : null;
  const testSamples = Number(
    raw.test_samples ?? raw.n_test ?? cmTotal
  );

  return {
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    test_samples: Number.isFinite(testSamples) ? testSamples : null,
    tn: hasCM ? tn : null,
    fp: hasCM ? fp : null,
    fn: hasCM ? fn : null,
    tp: hasCM ? tp : null,
    threshold: Number.isFinite(Number(raw.threshold))
      ? Number(raw.threshold)
      : 0.5
  };
}

function App() {
  const [rows, setRows] = useState(DEMO);
  const [fileName, setFileName] = useState("Dữ liệu demo minh họa");
  const [page, setPage] = useState("Overview");
  const [jobFilter, setJobFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [notice, setNotice] = useState(
    "Đang hiển thị dữ liệu demo minh họa. Tải CSV lên để phân tích dữ liệu của bạn."
  );

  const [predictForm, setPredictForm] = useState({
    age: 35,
    job: "management",
    marital: "married",
    education: "tertiary",
    balance: 1500,
    housing: "no",
    loan: "no",
    contact: "cellular",
    month: "may",
    campaign: 1,
    pdays: 999,
    previous: 0,
    poutcome: "unknown"
  });

  const [prediction, setPrediction] = useState(null);
  const [predictError, setPredictError] = useState("");
  const [metricsError, setMetricsError] = useState("");
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [modelMetrics, setModelMetrics] = useState(null);

  const apiBase = (import.meta.env.VITE_API_BASE_URL || "").trim().replace(/\/+$/, "");

  // Chỉ hiển thị metrics thực sự lấy được từ backend.
  useEffect(() => {
    let cancelled = false;

    async function loadMetrics() {
      setModelMetrics(null);
      setMetricsError("");

      if (!apiBase) {
        setMetricsError(
          "Chưa cấu hình VITE_API_BASE_URL. Metrics chưa được tải từ backend."
        );
        return;
      }

      setMetricsLoading(true);

      try {
        const response = await fetch(`${apiBase}/metrics`);

        if (!response.ok) {
          throw new Error(`API trả về HTTP ${response.status}`);
        }

        const raw = await response.json();
        const normalized = normalizeMetrics(raw);

        if (!normalized) {
          throw new Error(
            "Dữ liệu metrics không đủ hoặc không đúng định dạng mong đợi."
          );
        }

        if (!cancelled) {
          setModelMetrics(normalized);
          setMetricsError("");
        }
      } catch (error) {
        if (!cancelled) {
          setModelMetrics(null);
          setMetricsError(
            `Không tải được metrics: ${error.message}`
          );
        }
      } finally {
        if (!cancelled) setMetricsLoading(false);
      }
    }

    loadMetrics();

    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  const summary = useMemo(() => {
    const yes = rows.filter(r => normYes(r.y)).length;
    const total = rows.length;
    const jobs = [...new Set(
      rows.map(r => String(r.job || "unknown"))
    )].sort();

    const jobData = jobs.map(job => {
      const group = rows.filter(
        r => String(r.job || "unknown") === job
      );
      const subscribed = group.filter(r => normYes(r.y)).length;

      return {
        job,
        customers: group.length,
        rate: group.length ? subscribed / group.length : 0
      };
    }).sort((a, b) => b.customers - a.customers).slice(0, 7);

    const monthOrder = [
      "jan", "feb", "mar", "apr", "may", "jun",
      "jul", "aug", "sep", "oct", "nov", "dec"
    ];

    const monthData = monthOrder.map(month => {
      const group = rows.filter(
        r => String(r.month || "").toLowerCase() === month
      );

      return {
        month: month.toUpperCase(),
        customers: group.length,
        subscribed: group.filter(r => normYes(r.y)).length
      };
    }).filter(x => x.customers > 0);

    const ageGroups = [
      ["18–29", 18, 29],
      ["30–39", 30, 39],
      ["40–49", 40, 49],
      ["50–59", 50, 59],
      ["60+", 60, 200]
    ];

    const ageData = ageGroups.map(([label, min, max]) => {
      const group = rows.filter(
        r => safeNum(r.age) >= min && safeNum(r.age) <= max
      );

      return {
        group: label,
        customers: group.length,
        rate: group.length
          ? group.filter(r => normYes(r.y)).length / group.length
          : 0
      };
    }).filter(x => x.customers > 0);

    const outcomes = [
      { name: "Đăng ký", value: yes },
      { name: "Chưa đăng ký", value: total - yes }
    ];

    return {
      total,
      yes,
      no: total - yes,
      rate: total ? yes / total : 0,
      avgAge: total
        ? rows.reduce((s, r) => s + safeNum(r.age), 0) / total
        : 0,
      avgBalance: total
        ? rows.reduce((s, r) => s + safeNum(r.balance), 0) / total
        : 0,
      jobs,
      jobData,
      monthData,
      ageData,
      outcomes
    };
  }, [rows]);

  const shown = useMemo(() => {
    const term = search.toLowerCase();

    return rows.filter(r =>
      (jobFilter === "All" || String(r.job) === jobFilter) &&
      Object.values(r).some(v =>
        String(v ?? "").toLowerCase().includes(term)
      )
    ).slice(0, 100);
  }, [rows, jobFilter, search]);

  function uploadFile(file) {
    if (!file) return;

    setNotice("Đang kiểm tra tệp CSV...");
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,

      complete: result => {
        if (result.errors?.length) {
          setNotice(
            `CSV có ${result.errors.length} cảnh báo khi đọc. Hãy kiểm tra định dạng tệp trước khi sử dụng.`
          );
          return;
        }

        const cleaned = result.data
          .filter(r => r && Object.keys(r).length)
          .map(row => {
            const normalized = {};
            Object.entries(row).forEach(([key, value]) => {
              const cleanKey = key.replace(/^\uFEFF/, "").trim().toLowerCase();
              normalized[cleanKey] = value;
            });
            return normalized;
          });

        if (!cleaned.length) {
          setNotice("Không có dữ liệu. Hãy kiểm tra CSV có hàng tiêu đề.");
          return;
        }

        const missingColumns = REQUIRED_COLUMNS.filter(
          col => !Object.prototype.hasOwnProperty.call(cleaned[0], col)
        );

        if (missingColumns.length) {
          setNotice(
            `CSV chưa đúng cấu trúc Bank Marketing. Thiếu cột: ${missingColumns.join(", ")}.`
          );
          return;
        }

        const invalidTargets = cleaned.filter(row => {
          const target = String(row.y ?? "").trim().toLowerCase();
          return !["yes", "no", "1", "0"].includes(target);
        });

        if (invalidTargets.length) {
          setNotice(
            `Có ${invalidTargets.length} dòng có giá trị y không hợp lệ. Chỉ chấp nhận yes/no hoặc 1/0.`
          );
          return;
        }

        const normalizedRows = cleaned.map(row => ({
          ...row,
          y: ["yes", "1"].includes(String(row.y).trim().toLowerCase())
            ? "yes"
            : "no"
        }));

        setRows(normalizedRows);
        setFileName(file.name);
        setJobFilter("All");
        setSearch("");
        setPrediction(null);
        setNotice(
          `Đã nạp ${fmt(normalizedRows.length)} dòng từ ${file.name}. KPI và biểu đồ phân tích dựa trên tệp này; metrics mô hình vẫn lấy từ backend.`
        );
      },

      error: () => {
        setNotice("Không thể đọc tệp CSV. Vui lòng thử lại.");
      }
    });
  }

  function resetDemo() {
    setRows(DEMO);
    setFileName("Dữ liệu demo minh họa");
    setNotice(
      "Đang hiển thị dữ liệu demo minh họa, không phải kết quả nghiên cứu chính thức."
    );
    setJobFilter("All");
    setSearch("");
    setPrediction(null);
  }

  function downloadCsv() {
    const blob = new Blob(
      [Papa.unparse(rows)],
      { type: "text/csv;charset=utf-8;" }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "bank_marketing_data.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  async function predict() {
    setPredictError("");
    setPrediction(null);

    if (!apiBase) {
      setPredictError(
        "Chưa kết nối API. Hãy triển khai backend FastAPI và cấu hình VITE_API_BASE_URL."
      );
      return;
    }

    const age = Number(predictForm.age);
    const campaign = Number(predictForm.campaign);
    const pdays = Number(predictForm.pdays);
    const previous = Number(predictForm.previous);
    const balance = Number(predictForm.balance);

    if (!Number.isFinite(age) || age < 18 || age > 120) {
      setPredictError("Tuổi phải nằm trong khoảng 18–120.");
      return;
    }

    if (
      !Number.isFinite(campaign) || campaign < 0 ||
      !Number.isFinite(pdays) || pdays < -1 ||
      !Number.isFinite(previous) || previous < 0 ||
      !Number.isFinite(balance)
    ) {
      setPredictError("Vui lòng kiểm tra lại các trường số.");
      return;
    }

    try {
      const response = await fetch(`${apiBase}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...predictForm,
          age,
          balance,
          campaign,
          pdays,
          previous
        })
      });

      if (!response.ok) {
        let detail = "";
        try {
          const body = await response.json();
          detail = body.detail
            ? `: ${typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)}`
            : "";
        } catch {
          // Phản hồi không phải JSON.
        }

        throw new Error(`API trả về HTTP ${response.status}${detail}`);
      }

      const data = await response.json();
      const probability = Number(data.probability);

      if (
        !["yes", "no"].includes(data.prediction) ||
        !Number.isFinite(probability) ||
        probability < 0 ||
        probability > 1
      ) {
        throw new Error("Phản hồi dự đoán không đúng định dạng.");
      }

      setPrediction(data);
    } catch (error) {
      setPredictError(
        `Không gọi được API dự đoán: ${error.message}. Kiểm tra URL backend, CORS và trạng thái model.`
      );
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Activity size={21} /></div>
          <div>
            <strong>bankwise</strong>
            <span>MARKETING INTELLIGENCE</span>
          </div>
        </div>

        <div className="workspace">
          <span className="workspace-dot" />
          PROJECT WORKSPACE <ChevronDown size={14} />
        </div>

        <div className="nav-label">ANALYTICS</div>

        {[
          ["Overview", LayoutDashboard],
          ["Customer insights", Users],
          ["Data explorer", Database],
          ["AI prediction", BrainCircuit]
        ].map(([name, Icon]) => (
          <button
            key={name}
            className={`nav-item ${page === name ? "active" : ""}`}
            onClick={() => setPage(name)}
          >
            <Icon size={18} />
            <span>{name}</span>
            {name === "AI prediction" && <span className="new-pill">ML</span>}
          </button>
        ))}

        <div className="sidebar-bottom">
          <div className="status-icon"><ShieldCheck size={17} /></div>
          <div>
            <b>Academic project</b>
            <small>Bank Marketing · UCI dataset</small>
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="crumb">
            <span>Workspace</span><span>/</span><b>{page}</b>
          </div>
          <div className="top-actions">
            <span className="live-dot" />
            <span className="data-label">Local data session</span>
            <div className="avatar">BM</div>
          </div>
        </header>

        <div className="content">
          <section className="page-heading">
            <div>
              <div className="eyebrow">BANK MARKETING / ANALYTICS</div>
              <h1>
                {page === "Overview"
                  ? "Campaign overview"
                  : page === "Customer insights"
                    ? "Customer insights"
                    : page === "Data explorer"
                      ? "Explore your data"
                      : "Subscription prediction"}
              </h1>
              <p>
                {page === "Overview"
                  ? "Theo dõi hiệu quả chiến dịch và hành vi khách hàng trên một giao diện."
                  : page === "Customer insights"
                    ? "Khám phá khác biệt giữa các nhóm khách hàng và kết quả đăng ký."
                    : page === "Data explorer"
                      ? "Tải lên, tìm kiếm và kiểm tra dữ liệu phục vụ phân tích."
                      : "Dự đoán bằng mô hình machine learning đã huấn luyện từ dữ liệu UCI."}
              </p>
            </div>

            <label className="upload-btn">
              <FileUp size={16} /> Upload CSV
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={e => {
                  uploadFile(e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
            </label>
          </section>

          <div className="notice">
            <CircleHelp size={16} />
            <span>{notice}</span>
            <button onClick={resetDemo}>Reset demo</button>
          </div>

          <div className="source-line">
            <span className="source-dot" />
            <span>Current source: <b>{fileName}</b></span>
            <span className="source-sep">·</span>
            <span>{fmt(summary.total)} records</span>
            <span className="source-sep">·</span>
            <span>{Object.keys(rows[0] || {}).length} fields detected</span>
          </div>

          {page === "Overview" && (
            <>
              <div className="kpi-grid">
                <Kpi icon={Users} label="Total customers" value={fmt(summary.total)} foot="Bản ghi đang phân tích" color="purple" />
                <Kpi icon={Gauge} label="Subscription rate" value={pct(summary.rate)} foot={`${fmt(summary.yes)} khách hàng đăng ký`} color="green" />
                <Kpi icon={WalletCards} label="Average balance" value={Math.round(summary.avgBalance).toLocaleString("vi-VN")} foot="Đơn vị theo dữ liệu nguồn" color="amber" />
                <Kpi icon={Activity} label="Average age" value={`${summary.avgAge.toFixed(1)} yrs`} foot="Tuổi trung bình" color="blue" />
              </div>

              <div className="chart-grid">
                <Panel title="Subscription outcome" subtitle="Phân bố nhãn mục tiêu y" extra="TARGET">
                  <div className="donut-wrap">
                    <ResponsiveContainer width="55%" height={220}>
                      <PieChart>
                        <Pie data={summary.outcomes} dataKey="value" nameKey="name" innerRadius={62} outerRadius={86} paddingAngle={4}>
                          <Cell fill="#6f7df4" />
                          <Cell fill="#dce2ee" />
                        </Pie>
                        <Tooltip formatter={v => fmt(v)} />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="donut-center">
                      <strong>{pct(summary.rate)}</strong>
                      <span>registered</span>
                    </div>
                    <div className="legend">
                      {summary.outcomes.map((x, i) => (
                        <div key={x.name}>
                          <i style={{ background: i ? "#dce2ee" : "#6f7df4" }} />
                          {x.name}<b>{fmt(x.value)}</b>
                        </div>
                      ))}
                    </div>
                  </div>
                </Panel>

                <Panel title="Customer volume by job" subtitle="Nhóm nghề nghiệp có nhiều bản ghi nhất" extra="TOP 7">
                  <ResponsiveContainer width="100%" height={245}>
                    <BarChart data={summary.jobData} margin={{ top: 12, right: 8, left: -16, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#edf0f6" />
                      <XAxis dataKey="job" tick={{ fontSize: 10, fill: "#748096" }} axisLine={false} tickLine={false} interval={0} angle={-15} textAnchor="end" height={48} />
                      <YAxis tick={{ fontSize: 11, fill: "#748096" }} axisLine={false} tickLine={false} />
                      <Tooltip />
                      <Bar dataKey="customers" name="Customers" fill="#7c8cff" radius={[5, 5, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Panel>

                <Panel title="Subscription by age group" subtitle="Tỷ lệ đăng ký theo nhóm tuổi" extra="AGE">
                  <ResponsiveContainer width="100%" height={245}>
                    <BarChart data={summary.ageData} margin={{ top: 12, right: 8, left: -12, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#edf0f6" />
                      <XAxis dataKey="group" tick={{ fontSize: 11, fill: "#748096" }} axisLine={false} tickLine={false} />
                      <YAxis tickFormatter={v => `${Math.round(v * 100)}%`} domain={[0, 1]} tick={{ fontSize: 11, fill: "#748096" }} axisLine={false} tickLine={false} />
                      <Tooltip formatter={v => pct(v)} />
                      <Bar dataKey="rate" name="Subscription rate" fill="#34c89a" radius={[5, 5, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Panel>

                <Panel title="Campaign activity by month" subtitle="Số khách hàng theo tháng liên hệ" extra="MONTH">
                  <ResponsiveContainer width="100%" height={245}>
                    <LineChart data={summary.monthData} margin={{ top: 12, right: 12, left: -12, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#edf0f6" />
                      <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#748096" }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 11, fill: "#748096" }} axisLine={false} tickLine={false} />
                      <Tooltip />
                      <Line type="monotone" dataKey="customers" name="Customers" stroke="#7c8cff" strokeWidth={3} dot={{ r: 3, fill: "#7c8cff" }} />
                    </LineChart>
                  </ResponsiveContainer>
                </Panel>
              </div>

              <Panel title="Recent customer records" subtitle="Bản xem trước dữ liệu đang nạp" extra="PREVIEW">
                <DataTable rows={rows.slice(0, 6)} />
              </Panel>
            </>
          )}

          {page === "Customer insights" && (
            <>
              <div className="kpi-grid">
                <Kpi icon={Users} label="Customer segments" value={summary.jobs.length} foot="Nhãn nghề nghiệp khác nhau" color="purple" />
                <Kpi icon={Activity} label="Subscribed" value={fmt(summary.yes)} foot="Theo cột y trong dữ liệu" color="green" />
                <Kpi icon={WalletCards} label="Not subscribed" value={fmt(summary.no)} foot="Theo cột y trong dữ liệu" color="amber" />
                <Kpi icon={BarChart3} label="Conversion rate" value={pct(summary.rate)} foot="Tỷ lệ trên dữ liệu hiện tại" color="blue" />
              </div>

              <div className="chart-grid">
                <Panel title="Registration rate by job" subtitle="Tỷ lệ đăng ký trong từng nhóm nghề nghiệp" extra="SEGMENT">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={[...summary.jobData].sort((a, b) => b.rate - a.rate)} margin={{ top: 10, right: 10, left: 0, bottom: 35 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#edf0f6" />
                      <XAxis dataKey="job" angle={-25} textAnchor="end" interval={0} height={65} tick={{ fontSize: 10 }} />
                      <YAxis domain={[0, 1]} tickFormatter={v => `${Math.round(v * 100)}%`} />
                      <Tooltip formatter={v => pct(v)} />
                      <Bar dataKey="rate" name="Subscription rate" fill="#34c89a" radius={[5, 5, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Panel>

                <Panel title="Customer count by age" subtitle="Số lượng bản ghi trong từng nhóm tuổi" extra="DEMOGRAPHICS">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={summary.ageData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#edf0f6" />
                      <XAxis dataKey="group" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="customers" fill="#7c8cff" radius={[5, 5, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Panel>
              </div>

              <Panel title="Interpretation notes" subtitle="Cách sử dụng kết quả đúng trong báo cáo">
                <ul className="insight-list">
                  <li>Tỷ lệ đăng ký được tính từ cột mục tiêu <code>y</code> trong dữ liệu đang nạp.</li>
                  <li>Khác biệt giữa các nhóm phản ánh mối liên hệ quan sát được, không tự động chứng minh quan hệ nhân quả.</li>
                  <li>Model dự báo trước cuộc gọi chỉ tránh data leakage nếu <code>duration</code> thực sự bị loại trong bước huấn luyện.</li>
                  <li>Dữ liệu demo chỉ dùng để minh họa giao diện, không dùng làm kết quả nghiên cứu chính thức.</li>
                </ul>
              </Panel>
            </>
          )}

          {page === "Data explorer" && (
            <>
              <div className="kpi-grid">
                <Kpi icon={Database} label="Rows loaded" value={fmt(rows.length)} foot="Bản ghi hiện tại" color="purple" />
                <Kpi icon={BarChart3} label="Columns" value={Object.keys(rows[0] || {}).length} foot="Số trường phát hiện" color="blue" />
                <Kpi icon={Search} label="Rows shown" value={fmt(shown.length)} foot="Tối đa 100 dòng xem trước" color="green" />
                <Kpi icon={ShieldCheck} label="Duplicate rows" value={fmt(rows.length - new Set(rows.map(r => JSON.stringify(r))).size)} foot="Trùng toàn bộ trường" color="amber" />
              </div>

              <Panel title="Data preview" subtitle="Tìm kiếm theo nội dung và lọc nghề nghiệp">
                <div className="table-tools">
                  <div className="searchbox">
                    <Search size={16} />
                    <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search across fields..." />
                  </div>
                  <div className="selectbox">
                    <Filter size={15} />
                    <select value={jobFilter} onChange={e => setJobFilter(e.target.value)}>
                      <option value="All">All job types</option>
                      {summary.jobs.map(j => <option key={j} value={j}>{j}</option>)}
                    </select>
                  </div>
                  <button className="outline-btn" onClick={downloadCsv}>
                    <Download size={15} /> Export CSV
                  </button>
                </div>
                <DataTable rows={shown} />
                <div className="table-footer">
                  Hiển thị {shown.length} dòng phù hợp · Tệp nguồn không bị thay đổi
                </div>
              </Panel>

              <Panel title="Data quality checks" subtitle="Kiểm tra nhanh trên các trường đang có">
                <div className="quality-grid">
                  {[
                    {
                      label: "Missing values",
                      value: rows.reduce((n, r) => n + Object.values(r).filter(v => v === null || v === undefined || v === "").length, 0),
                      desc: "Giá trị null hoặc rỗng"
                    },
                    {
                      label: '"unknown" labels',
                      value: rows.reduce((n, r) => n + Object.values(r).filter(v => String(v).toLowerCase() === "unknown").length, 0),
                      desc: "Nhãn unknown được đếm riêng"
                    },
                    {
                      label: "Exact duplicate rows",
                      value: rows.length - new Set(rows.map(r => JSON.stringify(r))).size,
                      desc: "Trùng toàn bộ trường"
                    }
                  ].map(item => (
                    <div className="quality-card" key={item.label}>
                      <span>{item.label}</span>
                      <strong>{fmt(item.value)}</strong>
                      <small>{item.desc}</small>
                    </div>
                  ))}
                </div>
              </Panel>
            </>
          )}

          {page === "AI prediction" && (
            <>
              <div className="prediction-banner">
                <div className="prediction-icon"><BrainCircuit size={25} /></div>
                <div>
                  <b>Machine learning inference</b>
                  <p>
                    Form này gửi dữ liệu khách hàng đến model đã huấn luyện từ UCI Bank Marketing.
                    CSV tải lên chỉ phục vụ phân tích dữ liệu trên dashboard.
                  </p>
                </div>
                <span className={`api-status ${apiBase ? "connected" : "disconnected"}`}>
                  {apiBase ? "API configured" : "API not configured"}
                </span>
              </div>

              <div className="predict-layout">
                <Panel title="Customer profile" subtitle="Nhập thông tin trước chiến dịch liên hệ">
                  <div className="form-grid">
                    {[
                      ["age", "Age", "number"],
                      ["job", "Job", "selectjob"],
                      ["marital", "Marital status", "selectmarital"],
                      ["education", "Education", "selecteducation"],
                      ["balance", "Balance", "number"],
                      ["housing", "Housing loan", "yesno"],
                      ["loan", "Personal loan", "yesno"],
                      ["contact", "Contact type", "selectcontact"],
                      ["month", "Contact month", "selectmonth"],
                      ["campaign", "Contacts in campaign", "number"],
                      ["pdays", "Days since previous contact", "number"],
                      ["previous", "Previous contacts", "number"],
                      ["poutcome", "Previous outcome", "selectoutcome"]
                    ].map(([key, label, type]) => (
                      <label className="field" key={key}>
                        <span>{label}</span>

                        {type === "number" ? (
                          <input
                            type="number"
                            value={predictForm[key]}
                            min={
                              key === "age" ? 18 :
                                ["campaign", "previous"].includes(key) ? 0 :
                                  key === "pdays" ? -1 : undefined
                            }
                            max={key === "age" ? 120 : undefined}
                            onChange={e => {
                              const value = e.target.value;
                              setPredictForm(old => ({
                                ...old,
                                [key]: value === "" ? "" : Number(value)
                              }));
                            }}
                          />
                        ) : (
                          <select
                            value={predictForm[key]}
                            onChange={e => setPredictForm(old => ({
                              ...old,
                              [key]: e.target.value
                            }))}
                          >
                            {(type === "selectjob"
                              ? ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"]
                              : type === "selectmarital"
                                ? ["married", "single", "divorced"]
                                : type === "selecteducation"
                                  ? ["primary", "secondary", "tertiary", "unknown"]
                                  : type === "yesno"
                                    ? ["no", "yes"]
                                    : type === "selectcontact"
                                      ? ["cellular", "telephone", "unknown"]
                                      : type === "selectmonth"
                                        ? ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
                                        : ["failure", "other", "success", "unknown"]
                            ).map(value => <option key={value} value={value}>{value}</option>)}
                          </select>
                        )}
                      </label>
                    ))}
                  </div>

                  <button className="primary-btn predict-button" onClick={predict}>
                    <BrainCircuit size={17} /> Predict subscription likelihood
                  </button>

                  {predictError && <div className="error-box">{predictError}</div>}
                </Panel>

                <Panel title="Prediction result" subtitle="Kết quả trả về từ backend FastAPI">
                  <div className="result-placeholder">
                    {prediction ? (
                      <>
                        <div className="result-ring">
                          <strong>{pct(prediction.probability)}</strong>
                        </div>
                        <h3>
                          {prediction.prediction === "yes"
                            ? "Likely to subscribe"
                            : "Unlikely to subscribe"}
                        </h3>
                        <p>
                          {prediction.prediction === "yes"
                            ? "Mô hình phân loại khách hàng vào nhóm có khả năng đăng ký."
                            : "Mô hình phân loại khách hàng vào nhóm chưa đăng ký."}
                        </p>
                        <small>
                          Ngưỡng phân loại: {pct(prediction.threshold ?? 0.5)}.
                          Xác suất mô hình ước lượng không bảo đảm kết quả thực tế.
                        </small>
                      </>
                    ) : (
                      <>
                        <div className="empty-brain"><BrainCircuit size={34} /></div>
                        <h3>Awaiting prediction</h3>
                        <p>Điền thông tin khách hàng và kết nối backend để xem kết quả.</p>
                        <small>Không tạo kết quả giả khi API chưa sẵn sàng.</small>
                      </>
                    )}
                  </div>
                </Panel>
              </div>

              <Panel
                title="Model evaluation"
                subtitle="Các chỉ số được tải trực tiếp từ endpoint /metrics của backend"
                extra="MODEL METRICS"
              >
                {metricsLoading ? (
                  <div className="empty-table">Đang tải metrics từ backend...</div>
                ) : metricsError ? (
                  <div className="error-box">
                    {metricsError}
                    <p>
                      Các chỉ số chưa được hiển thị vì chưa xác minh được dữ liệu
                      metrics. Hãy kiểm tra endpoint /metrics và model.joblib.
                    </p>
                  </div>
                ) : modelMetrics ? (
                  <>
                    <div className="eval-metrics-grid">
                      <MetricCard
                        label="ROC-AUC"
                        value={pct(modelMetrics.roc_auc)}
                        description="Khả năng phân biệt hai lớp trên nhiều ngưỡng."
                      />
                      <MetricCard
                        label="RECALL"
                        value={pct(modelMetrics.recall)}
                        description="Tỷ lệ khách hàng thuộc lớp yes được nhận diện."
                      />
                      <MetricCard
                        label="PRECISION"
                        value={pct(modelMetrics.precision)}
                        description="Tỷ lệ dự đoán yes thực sự thuộc lớp yes."
                      />
                      <MetricCard
                        label="F1-SCORE"
                        value={pct(modelMetrics.f1)}
                        description="Trung bình điều hòa của Precision và Recall."
                      />
                      <MetricCard
                        label="ACCURACY"
                        value={pct(modelMetrics.accuracy)}
                        description="Tỷ lệ dự đoán đúng trên toàn bộ tập test."
                      />
                    </div>

                    <div className="eval-detail-grid">
                      {[
                        modelMetrics.tn,
                        modelMetrics.fp,
                        modelMetrics.fn,
                        modelMetrics.tp
                      ].every(Number.isFinite) ? (
                        <div className="cm-wrapper">
                          <div className="cm-title">
                            <b>Ma trận nhầm lẫn (Confusion Matrix)</b>
                            <span>
                              Ngưỡng phân loại: {pct(modelMetrics.threshold)}
                            </span>
                          </div>

                          <div className="cm-table-box">
                            <div className="cm-header-row">
                              <div className="cm-corner">Thực tế \ Dự đoán</div>
                              <div className="cm-col-label">Dự đoán: KHÔNG (No)</div>
                              <div className="cm-col-label">Dự đoán: CÓ (Yes)</div>
                            </div>

                            <div className="cm-data-row">
                              <div className="cm-row-label">Thực tế: KHÔNG (No)</div>
                              <div className="cm-cell tn">
                                <span className="cm-tag">True Negative (TN)</span>
                                <strong>{fmt(modelMetrics.tn)}</strong>
                                <small>Dự đoán đúng lớp no</small>
                              </div>
                              <div className="cm-cell fp">
                                <span className="cm-tag">False Positive (FP)</span>
                                <strong>{fmt(modelMetrics.fp)}</strong>
                                <small>Dự đoán yes nhưng thực tế no</small>
                              </div>
                            </div>

                            <div className="cm-data-row">
                              <div className="cm-row-label">Thực tế: CÓ (Yes)</div>
                              <div className="cm-cell fn">
                                <span className="cm-tag">False Negative (FN)</span>
                                <strong>{fmt(modelMetrics.fn)}</strong>
                                <small>Bỏ sót khách thuộc lớp yes</small>
                              </div>
                              <div className="cm-cell tp">
                                <span className="cm-tag">True Positive (TP)</span>
                                <strong>{fmt(modelMetrics.tp)}</strong>
                                <small>Dự đoán đúng lớp yes</small>
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="quality-card">
                          <b>Confusion matrix chưa có dữ liệu</b>
                          <small>
                            Backend cần trả tn/fp/fn/tp hoặc confusion_matrix
                            dạng [[TN, FP], [FN, TP]].
                          </small>
                        </div>
                      )}

                      <div className="model-specs-box">
                        <b>Diễn giải phương pháp</b>
                        <ul className="spec-list">
                          <li>
                            <strong>Thuật toán:</strong> Random Forest theo cấu hình
                            trong script huấn luyện.
                          </li>
                          <li>
                            <strong>Mất cân bằng lớp:</strong> class_weight="balanced"
                            điều chỉnh trọng số lớp trong quá trình huấn luyện,
                            không bảo đảm Recall sẽ đạt tối ưu.
                          </li>
                          <li>
                            <strong>Đánh giá:</strong> chỉ số phải lấy từ tập test
                            độc lập được sử dụng trong train_model.py.
                          </li>
                          <li>
                            <strong>Data leakage:</strong> duration cần được loại
                            trong bước huấn luyện nếu mục tiêu là dự báo trước cuộc gọi.
                          </li>
                        </ul>
                      </div>
                    </div>

                    <div className="table-footer">
                      {modelMetrics.test_samples !== null
                        ? `Số mẫu test theo backend: ${fmt(modelMetrics.test_samples)}.`
                        : "Backend chưa cung cấp số lượng mẫu test."}
                      {" "}Các chỉ số được hiển thị theo dữ liệu backend trả về.
                    </div>
                  </>
                ) : (
                  <div className="empty-table">Chưa có metrics để hiển thị.</div>
                )}
              </Panel>
            </>
          )}

          <footer>
            <span>Bankwise Analytics <span className="footer-dot">·</span> Academic project</span>
            <span>UCI Bank Marketing dataset <span className="footer-dot">·</span> For educational use</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

function Kpi({ icon: Icon, label, value, foot, color }) {
  return (
    <div className="kpi-card">
      <div className="kpi-top">
        <span>{label}</span>
        <span className={`kpi-icon ${color}`}><Icon size={18} /></span>
      </div>
      <strong>{value}</strong>
      <small>{foot}</small>
      <div className="kpi-bottom">
        <span className={`trend ${color}`}>●</span>
        <span>Based on current dataset</span>
      </div>
    </div>
  );
}

function Panel({ title, subtitle, extra, children }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        {extra && <span className="panel-tag">{extra}</span>}
      </div>
      {children}
    </section>
  );
}

function MetricCard({ label, value, description }) {
  return (
    <div className="eval-metric-card">
      <div className="eval-metric-header">
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
      <small>{description}</small>
    </div>
  );
}

function DataTable({ rows }) {
  if (!rows.length) {
    return <div className="empty-table">Không có dữ liệu phù hợp.</div>;
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.slice(0, 9).map(column => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.slice(0, 9).map(column => (
                <td key={column}>
                  {column === "y" ? (
                    <span className={`target-pill ${normYes(row[column]) ? "yes" : "no"}`}>
                      {String(row[column])}
                    </span>
                  ) : String(row[column] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
