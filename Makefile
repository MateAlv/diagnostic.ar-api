.PHONY: logs-diagnosticar

logs-diagnosticar:
	@echo "📋 Latest extraction logs (DB only, terminal access)"
	@docker compose exec -T audit-db psql \
		-U $${AUDIT_DB_USER:-diagnostic_audit} \
		-d $${AUDIT_DB_NAME:-diagnostic_audit} \
		-c "SELECT created_at, request_hash, patient_locale, cache_hit, duration_ms, left(text_es_raw, 80) AS text_es, left(text_en, 80) AS text_en, COALESCE(jsonb_array_length(phenotypes_json), 0) AS hpo_count, error FROM extraction_requests ORDER BY created_at DESC LIMIT $${LOGS_LIMIT:-20};"
