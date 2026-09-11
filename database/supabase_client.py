from __future__ import annotations

from app.logging_setup import get_logger

logger = get_logger("database.supabase_client")


def make_supabase_client(url: str, service_key: str, enabled: bool):
    if not enabled:
        logger.info("Supabase disabled via config/env", extra={"extra_fields": {"event_type": "supabase_disabled"}})
        return None
    if not url or not service_key:
        logger.warning("SUPABASE_URL/SUPABASE_SERVICE_KEY not set -- running without persistence",
                        extra={"extra_fields": {"event_type": "supabase_not_configured"}})
        return None
    try:
        from supabase import create_client
        client = create_client(url, service_key)
        logger.info("Supabase client initialized", extra={"extra_fields": {"event_type": "supabase_connected"}})
        return client
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize Supabase client", exc_info=exc)
        return None
