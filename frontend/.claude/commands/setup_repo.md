# Setup Repository

Bootstrap the ComfyUI Frontend monorepo with all necessary dependencies and verification checks.

## Overview

This command will:

1. Install pnpm package manager (if not present)
2. Install all project dependencies
3. Verify the project builds successfully
4. Run unit tests to ensure functionality
5. Start development server to verify frontend boots correctly

## Prerequisites Check

First, let's verify the environment:

```bash
# Check Node.js version (should be >= 24)
node --version

# Check if we're in a git repository
git status
```

## Step 1: Install pnpm

```bash
# Check if pnpm is already installed
pnpm --version 2>/dev/null || {
  echo "Installing pnpm..."
  npm install -g pnpm
}

# Verify pnpm installation
pnpm --version
```

## Step 2: Install Dependencies

```bash
# Install all dependencies using pnpm
echo "Installing project dependencies..."
pnpm install

# Verify node_modules exists and has packages
ls -la node_modules | head -5
```

## Step 3: Verify Build

```bash
# Run TypeScript type checking
echo "Running TypeScript checks..."
pnpm typecheck

# Build the project
echo "Building project..."
pnpm build

# Verify dist folder was created
ls -la dist/
```

## Step 4: Run Unit Tests

```bash
# Run unit tests
echo "Running unit tests..."
pnpm test:unit

# If tests fail, show the output and stop
if [ $? -ne 0 ]; then
  echo "❌ Unit tests failed. Please fix failing tests before continuing."
  exit 1
fi

echo "✅ Unit tests passed successfully"
```

## Step 5: Verify Development Server

```bash
# Start development server in background
echo "Starting development server..."
pnpm dev &
SERVER_PID=$!

# Wait for server to start (check for port 5173 or similar)
echo "Waiting for server to start..."
sleep 10

# Check if server is running
if curl -s http://localhost:5173 > /dev/null 2>&1; then
  echo "✅ Development server started successfully at http://localhost:5173"

  # Kill the background server
  kill $SERVER_PID
  wait $SERVER_PID 2>/dev/null
else
  echo "❌ Development server failed to start or is not accessible"
  kill $SERVER_PID 2>/dev/null
  wait $SERVER_PID 2>/dev/null
  exit 1
fi
```

## Step 6: Final Verification

```bash
# Run linting to ensure code quality
echo "Running linter..."
pnpm lint

# Show project status
echo ""
echo "🎉 Repository setup complete!"
echo ""
echo "Available commands:"
echo "  pnpm dev          - Start development server"
echo "  pnpm build        - Build for production"
echo "  pnpm test:unit    - Run unit tests"
echo "  pnpm typecheck    - Run TypeScript checks"
echo "  pnpm lint         - Run ESLint"
echo "  pnpm format       - Format code with oxfmt"
echo ""
echo "Next steps:"
echo "1. Run 'pnpm dev' to start developing"
echo "2. Open http://localhost:5173 in your browser"
echo "3. Check README.md for additional setup instructions"
```

## Troubleshooting

If any step fails:

1. **pnpm installation fails**: Try using `curl -fsSL https://get.pnpm.io/install.sh | sh -`
2. **Dependencies fail to install**: Try clearing cache with `pnpm store prune` and retry
3. **Build fails**: Check for TypeScript errors and fix them first
4. **Tests fail**: Review test output and fix failing tests
5. **Dev server fails**: Check if port 5173 is already in use

## Manual Verification Steps

After running the setup, manually verify:

1. **Dependencies installed**: `ls node_modules | wc -l` should show many packages
2. **Build artifacts**: `ls dist/` should show built files
3. **Server accessible**: Open http://localhost:5173 in browser
4. **Hot reload works**: Edit a file and see changes reflect

## Environment Requirements

- Node.js >= 24
- Git repository
- Internet connection for package downloads
- Available ports (typically 5173 for dev server)
