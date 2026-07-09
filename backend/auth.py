# Admin HTTP Basic Auth for the /dashboard/stats endpoint.
#
# Note: HTTP Basic Auth credentials are only base64-encoded, not encrypted —
# that's the HTTP spec, not something we can change here. Their security
# depends entirely on the connection being HTTPS. verify_admin() below refuses
# admin logins over plain HTTP (outside localhost) so credentials can never be
# sent in the clear, even if HTTPS is accidentally misconfigured later.

import os
import secrets
import logging
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

logger = logging.getLogger(__name__)

security = HTTPBasic()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(12)
    logger.warning(
        "ADMIN_PASSWORD env var not set! Generated a temporary random password "
        "for this session only (will change on every restart): %s\n"
        "Set ADMIN_USERNAME and ADMIN_PASSWORD as environment variables on your "
        "host for a permanent, secure login.",
        ADMIN_PASSWORD
    )


def verify_admin(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    # Render/most PaaS hosts terminate HTTPS at a proxy and forward plain HTTP
    # internally, so we check X-Forwarded-Proto (set by the proxy) rather than
    # request.url.scheme, which would always say "http" behind such a proxy.
    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    is_local = request.client and request.client.host in ("127.0.0.1", "localhost", "testclient")

    if forwarded_proto != "https" and not is_local:
        raise HTTPException(
            status_code=403,
            detail="Admin login requires a secure (HTTPS) connection."
        )

    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Invalid credentials",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials.username