import { useMemo, useState } from "react";

type Phenotype = {
  hpo_id: string;
  label: string;
  span_es: string;
  span_en: string;
  start_es: number;
  end_es: number;
  start_en: number;
  end_en: number;
  negated: boolean;
  confidence: number;
  matched_by: string;
};

type ExtractResponse = {
  text_en: string;
  model: {
    translation: string;
    phenotyper: string;
  };
  phenotypes: Phenotype[];
};

type SelectedPhenotype = Phenotype & { manual?: boolean };

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<ExtractResponse | null>(null);
  const [selected, setSelected] = useState<SelectedPhenotype[]>([]);
  const [manualTerm, setManualTerm] = useState("");

  const summary = useMemo(() => {
    if (!response) return null;
    return `${response.phenotypes.length} candidatos • ${response.model.translation}`;
  }, [response]);

  const handleExtract = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/extract-hpo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text_es: text, patient_locale: "es-AR" }),
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data: ExtractResponse = await res.json();
      setResponse(data);
      setSelected(data.phenotypes.map((item) => ({ ...item })));
    } catch (err) {
      setError("No se pudo procesar el texto. Reintentá en unos segundos.");
    } finally {
      setLoading(false);
    }
  };

  const removeItem = (hpoId: string) => {
    setSelected((prev) => prev.filter((item) => item.hpo_id !== hpoId));
  };

  const addManual = () => {
    const value = manualTerm.trim();
    if (!value) return;
    const id = `manual-${value.toLowerCase().replace(/\s+/g, "-")}`;
    if (selected.some((item) => item.hpo_id === id)) {
      setManualTerm("");
      return;
    }
    setSelected((prev) => [
      ...prev,
      {
        hpo_id: id,
        label: value,
        span_es: value,
        span_en: value,
        start_es: -1,
        end_es: -1,
        start_en: -1,
        end_en: -1,
        negated: false,
        confidence: 0.5,
        matched_by: "manual",
        manual: true,
      },
    ]);
    setManualTerm("");
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Diagnostic.ar · HPO Extractor</p>
          <h1>Convertí historias clínicas en HPOs explicables</h1>
          <p className="subtitle">
            Pegá texto en español y obtené términos HPO con spans, negaciones y
            confianza. Ajustá el resultado antes de seguir.
          </p>
        </div>
        <div className="hero-card">
          <p>Pipeline local, 100% open-source</p>
          <ul>
            <li>Traducción NLLB en GPU (si disponible)</li>
            <li>Matching HPO con sinónimos y fuzzy</li>
            <li>Sin logs de texto por defecto</li>
          </ul>
        </div>
      </header>

      <section className="panel">
        <div className="panel-header">
          <h2>Historia clínica</h2>
          <div className="panel-actions">
            <span className="hint">es-AR</span>
            <button
              className="primary"
              onClick={handleExtract}
              disabled={!text.trim() || loading}
            >
              {loading ? "Procesando..." : "Extraer HPO"}
            </button>
          </div>
        </div>
        <textarea
          placeholder="Ej: Paciente de 8 años con fiebre persistente, convulsiones nocturnas y sin dificultad respiratoria..."
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        {error && <p className="error">{error}</p>}
      </section>

      <section className="grid">
        <div className="panel">
          <div className="panel-header">
            <h2>Salida</h2>
            {summary && <span className="hint">{summary}</span>}
          </div>
          <div className="output">
            {response ? (
              <>
                <h3>Traducción EN</h3>
                <p className="translated">{response.text_en}</p>
                <h3>Fenotipos sugeridos</h3>
                <div className="chips">
                  {response.phenotypes.length === 0 && (
                    <span className="empty">Sin coincidencias con el índice HPO.</span>
                  )}
                  {response.phenotypes.map((item) => (
                    <span
                      key={item.hpo_id}
                      className={`chip ${item.negated ? "negated" : ""}`}
                      title={`Confianza ${item.confidence} · ${item.matched_by}`}
                    >
                      {item.label} · {item.hpo_id}
                    </span>
                  ))}
                </div>
              </>
            ) : (
              <p className="empty">Esperando texto para analizar.</p>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>Seleccionados</h2>
            <span className="hint">Editable</span>
          </div>
          <div className="chips">
            {selected.length === 0 && (
              <span className="empty">Todavía no hay HPOs seleccionados.</span>
            )}
            {selected.map((item) => (
              <button
                key={item.hpo_id}
                className={`chip removable ${item.negated ? "negated" : ""} ${
                  item.manual ? "manual" : ""
                }`}
                onClick={() => removeItem(item.hpo_id)}
                type="button"
              >
                {item.label}
                <span>×</span>
              </button>
            ))}
          </div>
          <div className="manual">
            <input
              value={manualTerm}
              onChange={(event) => setManualTerm(event.target.value)}
              placeholder="Agregar HPO manual"
            />
            <button className="ghost" onClick={addManual} type="button">
              Agregar
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
