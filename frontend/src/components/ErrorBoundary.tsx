import { Component } from 'react';
import type { ReactNode } from 'react';

/**
 * Props for ErrorBoundary component
 */
interface ErrorBoundaryProps {
  /** Child components to wrap */
  children: ReactNode;

  /** Optional fallback UI to show on error */
  fallback?: ReactNode;
}

/**
 * State for ErrorBoundary component
 */
interface ErrorBoundaryState {
  /** Whether an error has been caught */
  hasError: boolean;

  /** The error that was caught */
  error: Error | null;

  /** Additional error info from React */
  errorInfo: string | null;
}

/**
 * Error Boundary component to catch and handle React errors gracefully
 *
 * LEARNING REACT:
 * - Error boundaries MUST be class components (no hooks equivalent yet)
 * - They catch errors during:
 *   1. Rendering
 *   2. Lifecycle methods
 *   3. Constructors of child components
 * - They DON'T catch errors in:
 *   1. Event handlers (use try/catch)
 *   2. Async code (use try/catch)
 *   3. The error boundary itself
 *   4. Server-side rendering
 *
 * WHY USE THIS?
 * - Prevents entire app from crashing due to one component error
 * - Provides better UX with fallback UI
 * - Helps with debugging in production
 *
 * BEST PRACTICE:
 * - Wrap major sections of your app
 * - Don't wrap the entire app in ONE boundary (too coarse)
 * - Log errors to monitoring service (Sentry, LogRocket, etc.)
 *
 * @example
 * <ErrorBoundary>
 *   <MyComponent />
 * </ErrorBoundary>
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  /**
   * Static method called when an error is caught
   * Update state to trigger fallback UI
   */
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
    };
  }

  /**
   * Lifecycle method called after an error is caught
   * Use for logging, analytics, etc.
   */
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log error details
    console.error('[ErrorBoundary] Caught error:', error);
    console.error('[ErrorBoundary] Component stack:', errorInfo.componentStack);

    // Update state with error info
    this.setState({
      errorInfo: errorInfo.componentStack || null,
    });

    // TODO: Send to error tracking service
    // Example: Sentry.captureException(error, { extra: errorInfo });
  }

  /**
   * Reset error boundary state
   */
  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI
      return (
        <div className="min-h-screen w-full bg-black flex items-center justify-center p-6">
          <div className="max-w-2xl w-full bg-red-950/20 border border-red-500/30 rounded-2xl p-8">
            {/* Error icon */}
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-red-500/20 rounded-full flex items-center justify-center">
                <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <h1 className="text-2xl font-bold text-red-500">Something went wrong</h1>
            </div>

            {/* Error message */}
            <p className="text-gray-300 mb-6">
              The application encountered an unexpected error. This has been logged for investigation.
            </p>

            {/* Error details (development only) */}
            {import.meta.env.DEV && this.state.error && (
              <details className="mb-6">
                <summary className="text-sm text-gray-400 cursor-pointer hover:text-gray-300 mb-2">
                  Error Details (Development Only)
                </summary>
                <div className="bg-black/50 rounded-lg p-4 overflow-auto">
                  <p className="text-xs text-red-400 font-mono mb-2">
                    {this.state.error.toString()}
                  </p>
                  {this.state.errorInfo && (
                    <pre className="text-xs text-gray-500 overflow-x-auto">
                      {this.state.errorInfo}
                    </pre>
                  )}
                </div>
              </details>
            )}

            {/* Reset button */}
            <button
              onClick={this.handleReset}
              className="w-full px-4 py-3 bg-red-600 hover:bg-red-700 active:bg-red-800
                       text-white rounded-lg font-medium transition-all duration-150
                       focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-black"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    // No error, render children normally
    return this.props.children;
  }
}
