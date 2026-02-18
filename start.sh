#!/usr/bin/env bash
#
# Ragent - Docker Management Script
#
# Usage:
#   ./start.sh start    Start Qdrant + interactive CLI (auto-stop on exit) [default]
#   ./start.sh build    Build the app Docker image
#   ./start.sh down     Stop and remove all containers
#

set -e

COMPOSE="docker compose"
IMAGE_NAME="ragent"

case "${1:-start}" in

  build)
    echo ""
    echo "=========================================="
    echo "  Ragent - Building..."
    echo "=========================================="
    echo ""
    # Remove old image (if exists) to avoid dangling images after build
    docker image rm $IMAGE_NAME 2>/dev/null && echo "Removed old image '$IMAGE_NAME'." || true
    $COMPOSE build app
    echo ""
    echo "Build complete."
    ;;

  start)
    echo ""
    echo "=========================================="
    echo "  Ragent - Starting..."
    echo "=========================================="
    echo ""

    # Start Qdrant in background and wait for healthy
    echo "[1/2] Starting Qdrant..."
    $COMPOSE up qdrant -d --wait

    echo "[2/2] Launching Ragent CLI..."
    echo ""

    # Run app in interactive mode (foreground)
    $COMPOSE run --rm app

    # Cleanup when user exits
    echo ""
    echo "Stopping all containers..."
    $COMPOSE stop
    echo "Done. All containers stopped."
    ;;

  down)
    echo ""
    echo "=========================================="
    echo "  Ragent - Downing..."
    echo "=========================================="
    echo ""
    $COMPOSE down -v
    echo "Done. All containers downed."
    ;;

  *)
    echo "Usage: ./start.sh {start|build|down}"
    echo ""
    echo "  start    Start Qdrant + interactive CLI (auto-stop on exit) [default]"
    echo "  build    Build the app Docker image"
    echo "  down     Remove all containers"
    exit 1
    ;;

esac
