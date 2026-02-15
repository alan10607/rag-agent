#!/usr/bin/env bash
#
# VectorSearcher - Docker Management Script
#
# Usage:
#   ./start.sh build    Build the app Docker image
#   ./start.sh start    Start Qdrant + interactive CLI (auto-stop on exit) [default]
#   ./start.sh config   Show current configuration
#   ./start.sh down     Stop and remove all containers
#

set -e

COMPOSE="docker compose"
SERVICE="vector-searcher"

case "${1:-start}" in

  build)
    echo ""
    echo "=========================================="
    echo "  VectorSearcher - Building..."
    echo "=========================================="
    echo ""
    # Remove old image (if exists) to avoid dangling images after build
    docker image rm $SERVICE 2>/dev/null && echo "Removed old image '$SERVICE'." || true
    $COMPOSE build $SERVICE
    echo ""
    echo "Build complete."
    ;;

  start)
    echo ""
    echo "=========================================="
    echo "  VectorSearcher - Starting..."
    echo "=========================================="
    echo ""

    # Start Qdrant in background and wait for healthy
    echo "[1/2] Starting Qdrant..."
    $COMPOSE up qdrant -d --wait

    echo "[2/2] Launching VectorSearcher CLI..."
    echo ""

    # Run app in interactive mode (foreground)
    $COMPOSE run --rm $SERVICE

    # Cleanup when user exits
    echo ""
    echo "Stopping all containers..."
    $COMPOSE stop
    echo "Done. All containers stopped."
    ;;

  config)
    $COMPOSE run --rm --no-deps $SERVICE python -m app config
    ;;

  down)
    echo ""
    echo "=========================================="
    echo "  VectorSearcher - Downing..."
    echo "=========================================="
    echo ""
    $COMPOSE down -v
    echo "Done. All containers downed."
    ;;

  *)
    echo "Usage: ./start.sh {build|start|config|down}"
    echo ""
    echo "  build    Build the app Docker image"
    echo "  start    Start Qdrant + interactive CLI (auto-stop on exit) [default]"
    echo "  config   Show current configuration"
    echo "  down     Remove all containers"
    exit 1
    ;;

esac
