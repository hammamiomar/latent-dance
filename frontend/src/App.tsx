/**
 * App - Root component.
 *
 * Routes between browser mode (full-viewport visualizer) and
 * desktop mode (Computer Buddy character wrapping the visualizer).
 * IS_DESKTOP_MODE is set by ?desktop query param.
 */

import { IS_DESKTOP_MODE } from "./constants";
import { BrowserApp } from "./apps/BrowserApp";
import { BrainApp } from "./apps/BrainApp";
import { DesktopApp } from "./apps/DesktopApp";
import { ErrorBoundary } from "./components/ErrorBoundary";

function App() {
  const isBrainWindow = window.location.pathname === "/brain";
  return (
    <ErrorBoundary>
      {isBrainWindow ? <BrainApp /> : IS_DESKTOP_MODE ? <DesktopApp /> : <BrowserApp />}
    </ErrorBoundary>
  );
}

export default App;
