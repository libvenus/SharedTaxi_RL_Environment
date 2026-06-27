"""Out-of-process scheduled jobs.

Modules in this package are entry points for cron / Celery / scheduled task
runners. They are intentionally importable as both regular Python modules
and ``python -m app.jobs.<name>`` scripts so they can be wired into any
scheduler without modifying the FastAPI app.
"""
