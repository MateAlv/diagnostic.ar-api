.PHONY: logs-diagnosticar test test-quick info healthz up down logs

# Rich Spanish medical text with many detectable HPO symptoms
define TEST_MEDICAL_TEXT
Paciente masculino de 8 años de edad que consulta por cuadro de 3 días de evolución caracterizado por fiebre alta persistente de 39.5°C, cefalea intensa holocraneana, vómitos en proyectil y rigidez de nuca. \
La madre refiere que el niño presenta convulsiones tónico-clónicas generalizadas de aproximadamente 2 minutos de duración, con pérdida de conocimiento. \
Al examen físico se observa ictericia escleral leve, hepatomegalia palpable a 3 cm del reborde costal derecho y esplenomegalia. \
Presenta además dificultad respiratoria con tiraje intercostal, cianosis perioral y saturación de oxígeno del 88%. \
Se evidencia edema bipalpebral y edema de miembros inferiores con fóvea positiva. \
El paciente refiere dolor abdominal difuso tipo cólico, diarrea acuosa sin sangre y pérdida de peso de aproximadamente 4 kg en el último mes. \
Antecedentes: retraso del desarrollo psicomotor, hipotonía muscular desde el nacimiento, microcefalia y estrabismo convergente. \
La familia reporta episodios previos de hipoglucemia sintomática y sordera neurosensorial bilateral diagnosticada a los 2 años. \
Al examen neurológico presenta ataxia de la marcha, nistagmo horizontal, disartria y temblor intencional en miembros superiores.
endef
export TEST_MEDICAL_TEXT

API_URL ?= http://localhost:8000

test:
	@echo "Testing HPO extraction with comprehensive Spanish medical text..."
	@echo ""
	@echo "Input text:"
	@echo "$$TEST_MEDICAL_TEXT" | fold -s -w 100
	@echo ""
	@echo "Sending request to $(API_URL)/extract-hpo..."
	@echo ""
	@curl -sf --max-time 120 -X POST "$(API_URL)/extract-hpo" \
		-H "Content-Type: application/json" \
		-d "{\"text_es\": \"$$TEST_MEDICAL_TEXT\", \"patient_locale\": \"es-AR\"}" | \
		jq '.' || (echo "❌ Request failed or timed out. Is the API running? Try: make healthz" && exit 1)

test-quick:
	@echo "Quick test with short medical text..."
	@curl -sf -X POST "$(API_URL)/extract-hpo" \
		-H "Content-Type: application/json" \
		-d '{"text_es": "Paciente con fiebre alta, convulsiones, cefalea intensa y vómitos. Presenta ictericia, hepatomegalia y dificultad respiratoria.", "patient_locale": "es-AR"}' | \
		jq '.' || (echo "❌ Request failed. Is the API running? Try: make healthz" && exit 1)

info:
	@echo "API Info:"
	@curl -sf "$(API_URL)/info" | jq '.' || (echo "❌ Request failed. Is the API running?" && exit 1)

healthz:
	@echo "Checking API health at $(API_URL)..."
	@curl -sf "$(API_URL)/healthz" | jq '.' && echo "✅ API is healthy" || echo "❌ API is not responding at $(API_URL)"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f api

logs-diagnosticar:
	@echo "Latest extraction logs (DB only, terminal access)"
	@docker compose exec -T audit-db psql \
		-U $${AUDIT_DB_USER:-diagnostic_audit} \
		-d $${AUDIT_DB_NAME:-diagnostic_audit} \
		-c "SELECT created_at, request_hash, patient_locale, cache_hit, duration_ms, left(text_es_raw, 80) AS text_es, left(text_en, 80) AS text_en, COALESCE(jsonb_array_length(phenotypes_json), 0) AS hpo_count, error FROM extraction_requests ORDER BY created_at DESC LIMIT $${LOGS_LIMIT:-20};"
