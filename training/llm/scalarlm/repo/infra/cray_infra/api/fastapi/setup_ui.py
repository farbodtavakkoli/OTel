import logging
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)

UI_BUNDLE_DIR = os.environ.get("SCALARLM_UI_BUNDLE_DIR", "/app/ui-bundle")

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

    @app.get("/", include_in_schema=False)
    async def _root_redirect():
        return RedirectResponse(url="/app/", status_code=302)

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

    @app.get("/app/api-config.json", include_in_schema=False)
    async def _api_config():
        config = get_config()
        return {
            "api_base": "/v1",
            "version": VERSION,
            "default_model": config.get("model", ""),
            "features": {},
        }

    @app.get("/app/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str):
        return FileResponse(
            index_html,
            headers={"Cache-Control": "no-cache"},
        )
