#!/bin/bash
cd "$(dirname "$0")" || exit 1
exec .venv2/bin/python -m mcp_server
