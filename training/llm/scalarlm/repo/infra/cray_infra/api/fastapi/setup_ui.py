"""
Mount the ScalarLM React SPA onto the FastAPI app at /app/*.

Layout (built by `ui/` at image build time, copied into the final image):

    /app/ui-bundle/
        index.html
        assets/
            index-{hash}.js
            index-{hash}.css
            ...

Route plan — order matters; StaticFiles mount and specific routes must
register BEFORE the catch-all SPA fallback:

    GET /                       → 302 /app/
    GET /app/assets/*           → StaticFiles (immutable, 1y cache)
    GET /app/api-config.json    → runtime config JSON
    GET /app/{full_path:path}   → index.html (SPA history routing)

If the bundle directory does not exist on disk (e.g. backend-only dev image
without a UI build), the routes are not registered. Requests to /app/* then
fall through to 404, and the backend continues to work.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)

# Overridable for testing / development. In production the Dockerfile copies
# the built UI bundle to this exact path.
UI_BUNDLE_DIR = os.environ.get("SCALARLM_UI_BUNDLE_DIR", "/app/ui-bundle")

# Advertised to the frontend via /app/api-config.json so the version stays
# in sync with whatever release tagged the image. Not yet threaded through
# the Helm chart; overridable by env var meanwhile.
VERSION = os.environ.get("SCALARLM_VERSION", "dev")


def add_ui(app: FastAPI) -> None:
    index_html = os.path.join(UI_BUNDLE_DIR, "index.html")
    assets_dir = os.path.join(UI_BUNDLE_DIR, "assets")

    if not os.path.isfile(index_html):
        logger.warning(
            "ScalarLM UI bundle not found at %s; skipping /app/* routes. "
            "Build with `cd ui && npm run build` or rebuild the image.",
            UI_BUNDLE_DIR,
        )
        return

    logger.info("Mounting ScalarLM UI from %s", UI_BUNDLE_DIR)

    # Root -> /app/. Exact-match route, fine to register any time.
    @app.get("/", include_in_schema=False)
    async def _root_redirect():
        return RedirectResponse(url="/app/", status_code=302)

    # Hashed, content-addressed assets: cache forever. Mount BEFORE the
    # catch-all so Starlette matches the mount first.
    if os.path.isdir(assets_dir):
        app.mount(
            "/app/assets",
            StaticFiles(directory=assets_dir),
            name="scalarlm-ui-assets",
        )
    else:
        logger.warning(
            "UI bundle has no assets/ subdirectory at %s; serving index only.",
            assets_dir,
        )

    # Runtime configuration read once at app boot by the SPA. Kept minimal so
    # the endpoint is cheap to call and adding fields does not force a rebuild.
    @app.get("/app/api-config.json", include_in_schema=False)
    async def _api_config():
        config = get_config()
        return {
            "api_base": "/v1",
            "version": VERSION,
            "default_model": config.get("model", ""),
            "features": {},
        }

    # SPA history routing: any path under /app/ that isn't a static asset
    # returns index.html with no-cache so the latest asset manifest wins.
    # Registered LAST so the mount and api-config handler take precedence.
    @app.get("/app/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str):  # noqa: ARG001 — path unused
        return FileResponse(
            index_html,
            headers={"Cache-Control": "no-cache"},
        )
