#!/bin/bash
# Install git hooks from tracked hooks/ directory

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="$REPO_ROOT/hooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

echo "Installing git hooks..."

# Copy pre-push hook
if [ -f "$HOOKS_SRC/pre-push" ]; then
    cp "$HOOKS_SRC/pre-push" "$HOOKS_DST/pre-push"
    chmod +x "$HOOKS_DST/pre-push"
    echo "  ✓ pre-push hook installed"
else
    echo "  ✗ pre-push hook not found in hooks/"
    exit 1
fi

echo "Done. Hooks installed successfully."
exit 0
